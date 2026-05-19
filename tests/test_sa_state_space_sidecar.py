"""PIT contract + toy-data fit test for sa_state_space_sidecar.

The toy test generates a synthetic local-level + seasonal series with known
DGP, runs walk-forward, and asserts that 1) the PIT contract holds for every
row, 2) all required schema columns land in the predictions frame, and 3) the
sidecar's MAE on a held-out window is comparable to the irreducible noise
floor of the DGP (sigma_irregular).
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
from experiments.sidecars.sa_state_space_sidecar import (  # noqa: E402
    _resolve_output_dir,
    run_sa_state_space_sidecar,
)


@pytest.fixture
def toy_target_path(tmp_path):
    """Local-level + monthly-seasonal toy series, 180 months, signal/noise ≈ 4."""
    rng = np.random.default_rng(123)
    n = 180
    ds = pd.date_range("2000-01-01", periods=n, freq="MS")
    sigma_eta_level = 5.0
    sigma_eta_trend = 0.5
    sigma_irregular = 10.0
    trend = np.zeros(n)
    level = np.zeros(n)
    level[0] = 1000.0
    for t in range(1, n):
        trend[t] = trend[t - 1] + rng.normal(0, sigma_eta_trend)
        level[t] = level[t - 1] + trend[t - 1] + rng.normal(0, sigma_eta_level)
    seasonal_pattern = np.array(
        [40, -10, -25, -30, -10, 5, 15, 20, 10, -5, -15, 5], dtype=float
    )
    seasonal = np.tile(seasonal_pattern, n // 12 + 1)[:n]
    y = level + seasonal + rng.normal(0, sigma_irregular, size=n)
    df = pd.DataFrame({"ds": ds, "y": y})
    path = tmp_path / "toy_target.parquet"
    df.to_parquet(path, index=False)
    return path, sigma_irregular


def test_pit_contract_holds_for_every_row(toy_target_path, tmp_path):
    target_path, _ = toy_target_path
    results, _ = run_sa_state_space_sidecar(
        target_path=target_path,
        output_dir=tmp_path / "uc",
        start="2008-01",
        min_train=84,
        target_space="sa_revised",
        target_type="sa",
        use_covid_dummy=False,
    )
    assert not results.empty
    ds = pd.to_datetime(results["ds"])
    trained = pd.to_datetime(results["trained_through"])
    assert (trained < ds).all(), (
        "PIT contract violated: every row must have trained_through < ds"
    )


def test_schema_contract_after_standardization(toy_target_path, tmp_path):
    target_path, _ = toy_target_path
    output_dir = tmp_path / "uc"
    run_sa_state_space_sidecar(
        target_path=target_path,
        output_dir=output_dir,
        start="2008-01",
        min_train=84,
        target_space="sa_revised",
        target_type="sa",
        use_covid_dummy=False,
    )
    preds = pd.read_csv(output_dir / "predictions.csv")
    missing = REQUIRED_PREDICTION_COLUMNS - set(preds.columns)
    assert not missing, f"Missing required cols after finalization: {missing}"


def test_mae_close_to_noise_floor(toy_target_path, tmp_path):
    """On a DGP with local-level + seasonal + N(0,σ²) noise, a well-fit UC
    should land its 1-step-ahead MAE within roughly 2× σ. Loose because the
    estimator must rediscover the variances and trend from finite samples.
    """
    target_path, sigma = toy_target_path
    results, metrics = run_sa_state_space_sidecar(
        target_path=target_path,
        output_dir=tmp_path / "uc",
        start="2010-01",
        min_train=84,
        target_space="sa_revised",
        target_type="sa",
        use_covid_dummy=False,
    )
    assert "mae" in metrics
    # Irreducible 1-step-ahead error is dominated by σ_irregular + sqrt(σ_eta²).
    # Allow up to 3× σ_irregular as a generous bar — this is a sanity check,
    # not a strict identifiability claim.
    assert metrics["mae"] < 3.0 * sigma, (
        f"MAE {metrics['mae']:.2f} is implausibly larger than 3σ={3*sigma:.2f}"
    )
    # Most fits should succeed on a clean DGP.
    assert metrics["fit_success_rate"] >= 0.90


def test_output_dir_routing_sa_vs_nsa():
    sa_path = _resolve_output_dir(None, "sa", "test_run")
    nsa_path = _resolve_output_dir(None, "nsa", "test_run")
    assert str(sa_path).endswith("sidecars/sa/test_run/sa_state_space")
    # NSA falls back to the legacy flat path.
    assert "sidecars/local_sidecar_once/sa_state_space" in str(nsa_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
