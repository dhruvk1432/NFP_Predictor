import numpy as np
import pandas as pd

from experiments.sidecars.acceleration_classifier_sidecar import (
    _add_composites,
    run_acceleration_classifier_sidecar,
)
from experiments.sidecars.feature_matrix import add_target_dynamics


def test_sidecar_target_dynamics_exclude_same_day_revised_actuals():
    dates = pd.date_range("2021-01-01", periods=4, freq="MS")
    target = pd.DataFrame({
        "ds": dates,
        "y_mom": [10.0, 20.0, 30.0, 40.0],
        "release_date": dates + pd.DateOffset(months=1, days=4),
        "operational_available_date": dates + pd.DateOffset(months=2, days=4),
    })

    out = add_target_dynamics(target, prefix="sa")

    mar = out[out["ds"] == pd.Timestamp("2021-03-01")].iloc[0]
    apr = out[out["ds"] == pd.Timestamp("2021-04-01")].iloc[0]
    assert mar["prev_mom"] == 10.0
    assert mar["actual_accel"] == 20.0
    assert apr["prev_mom"] == 20.0
    assert apr["sa_mom_lag1"] == 20.0

    with_composites = _add_composites(out)
    apr_comp = with_composites[with_composites["ds"] == pd.Timestamp("2021-04-01")].iloc[0]
    assert apr_comp["composite_target_mom_accel_interaction"] == apr["prev_mom"] * apr["sa_accel_lag1"]


def test_acceleration_classifier_sidecar_writes_contract(tmp_path):
    rng = np.random.default_rng(123)
    dates = pd.date_range("2000-01-01", periods=150, freq="MS")
    mom = np.sin(np.arange(len(dates)) / 5.0) * 40 + rng.normal(0, 10, len(dates))
    level = 100_000 + np.cumsum(mom)
    target_path = tmp_path / "target.parquet"
    pd.DataFrame({"ds": dates, "y": level, "y_mom": mom}).to_parquet(target_path, index=False)

    results, metrics = run_acceleration_classifier_sidecar(
        target_path=target_path,
        output_dir=tmp_path / "accel",
        start="2004-01",
        min_train=36,
        include_snapshots=False,
        model_kind="logistic",
        top_features=12,
    )

    assert not results.empty
    assert metrics["n_predictions"] == len(results)
    assert 0.0 <= metrics["acceleration_accuracy"] <= 1.0
    assert (tmp_path / "accel" / "predictions.csv").exists()
    assert (tmp_path / "accel" / "metrics.json").exists()
