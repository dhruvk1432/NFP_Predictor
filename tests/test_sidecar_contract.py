import json

import pandas as pd
import pytest

from experiments.sidecars.common import validate_pit_predictions, write_sidecar_artifacts


def _predictions():
    ds = pd.date_range("2020-01-01", periods=4, freq="MS")
    return pd.DataFrame(
        {
            "ds": ds,
            "trained_through": ds - pd.offsets.MonthBegin(1),
            "predicted_mom": [10.0, -5.0, 3.0, 6.0],
            "predicted_accel": [2.0, -15.0, 8.0, 3.0],
            "predicted_accel_proba_up": [0.6, 0.2, 0.7, 0.8],
            "confidence": [0.2, 0.6, 0.4, 0.6],
            "uncertainty": [0.8, 0.4, 0.6, 0.4],
            "actual_mom": [8.0, -7.0, 2.0, 9.0],
            "actual_accel": [1.0, -15.0, 9.0, 7.0],
        }
    )


def test_sidecar_contract_writes_standard_artifacts(tmp_path):
    out_dir = tmp_path / "run" / "model"
    preds, metrics = write_sidecar_artifacts(
        output_dir=out_dir,
        model_id="contract_test",
        target_space="sa_revised",
        predictions=_predictions(),
        feature_audit=pd.DataFrame(
            [{"feature": "x", "source": "unit", "missing_rate": 0.0, "pit_cutoff": "trained_through"}]
        ),
        config={"unit": True},
        data_paths={"target": "synthetic"},
    )

    assert len(preds) == 4
    assert metrics["n_predictions"] == 4
    assert {"predictions.csv", "metrics.json", "feature_audit.csv", "manifest.json"} <= {
        p.name for p in out_dir.iterdir()
    }
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["pit_validation"] == "trained_through < ds"


def test_sidecar_contract_rejects_non_pit_rows():
    bad = _predictions()
    bad.loc[0, "trained_through"] = bad.loc[0, "ds"]
    standardized = bad.assign(
        model_id="bad",
        target_space="sa_revised",
        predicted_accel_sign=1.0,
    )
    with pytest.raises(ValueError, match="strictly before ds"):
        validate_pit_predictions(standardized)
