import numpy as np
import pandas as pd

from experiments.sidecars.acceleration_classifier_sidecar import run_acceleration_classifier_sidecar


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
