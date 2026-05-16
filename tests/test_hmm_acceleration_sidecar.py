import json

import numpy as np
import pandas as pd

from experiments.sidecars.hmm_acceleration_sidecar import run_hmm_acceleration_sidecar


def test_hmm_acceleration_sidecar_writes_standalone_metrics(tmp_path):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=140, freq="MS")
    mom = np.sin(np.arange(len(dates)) / 6.0) * 60 + rng.normal(0, 15, len(dates))
    level = 100_000 + np.cumsum(mom)
    target = pd.DataFrame({"ds": dates, "y": level, "y_mom": mom})
    target_path = tmp_path / "target.parquet"
    out_dir = tmp_path / "sidecar"
    target.to_parquet(target_path, index=False)

    results, metrics = run_hmm_acceleration_sidecar(
        target_path=target_path,
        output_dir=out_dir,
        start="2004-01",
        min_train=36,
        n_components=2,
        include_snapshots=False,
    )

    assert not results.empty
    assert metrics["n_predictions"] == len(results)
    assert 0.0 <= metrics["acceleration_accuracy"] <= 1.0
    assert (out_dir / "predictions.csv").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "feature_audit.csv").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "hmm_acceleration_predictions.csv").exists()
    saved_metrics = json.loads((out_dir / "hmm_acceleration_metrics.json").read_text())
    assert saved_metrics["n_predictions"] == len(results)
