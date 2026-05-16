import json

import numpy as np
import pandas as pd

from experiments.sidecars.integration import attach_sidecar_features, load_sidecar_feature_frame
from Train.Output_code.consensus_anchor_runner import kalman_fusion


def _write_sidecar(base, run_id="run_a", model_id="accel"):
    model_dir = base / "sidecars" / run_id / model_id
    model_dir.mkdir(parents=True)
    ds = pd.date_range("2021-01-01", periods=3, freq="MS")
    pd.DataFrame(
        {
            "ds": ds,
            "model_id": model_id,
            "target_space": "sa_revised",
            "predicted_mom": [1.0, 2.0, 3.0],
            "predicted_accel": [0.2, -0.1, 0.3],
            "predicted_accel_sign": [1.0, -1.0, 1.0],
            "predicted_accel_proba_up": [0.7, 0.4, 0.8],
            "confidence": [0.4, 0.2, 0.6],
            "uncertainty": [0.6, 0.8, 0.4],
            "trained_through": ds - pd.offsets.MonthBegin(1),
            "regime_transition_risk": [0.1, 0.2, 0.3],
        }
    ).to_csv(model_dir / "predictions.csv", index=False)
    (model_dir / "metrics.json").write_text(json.dumps({"promotion_gate_passed": False}))
    return model_dir


def test_sidecar_features_are_dormant_by_default(tmp_path, monkeypatch):
    _write_sidecar(tmp_path)
    X = pd.DataFrame({"ds": pd.date_range("2021-01-01", periods=3, freq="MS"), "x": [1, 2, 3]})

    monkeypatch.delenv("NFP_SIDECAR_MODE", raising=False)
    monkeypatch.delenv("NFP_ENABLE_SIDECAR_FEATURES", raising=False)
    out = attach_sidecar_features(X, output_dir=tmp_path)
    pd.testing.assert_frame_equal(out, X)


def test_sidecar_features_attach_when_explicitly_enabled(tmp_path, monkeypatch):
    _write_sidecar(tmp_path)
    X = pd.DataFrame({"ds": pd.date_range("2021-01-01", periods=3, freq="MS"), "x": [1, 2, 3]})
    monkeypatch.setenv("NFP_SIDECAR_MODE", "features")
    monkeypatch.setenv("NFP_SIDECAR_RUN_ID", "run_a")
    monkeypatch.setenv("NFP_SIDECAR_REQUIRE_PASSING_GATE", "0")

    out = attach_sidecar_features(X, output_dir=tmp_path)
    sidecar_cols = [c for c in out.columns if c.startswith("sidecar_accel__")]
    assert "sidecar_accel__predicted_mom" in sidecar_cols
    assert out["sidecar_accel__predicted_mom"].tolist() == [1.0, 2.0, 3.0]


def test_sidecar_gate_skips_non_passing_models(tmp_path, monkeypatch):
    _write_sidecar(tmp_path)
    monkeypatch.setenv("NFP_SIDECAR_RUN_ID", "run_a")
    monkeypatch.setenv("NFP_SIDECAR_REQUIRE_PASSING_GATE", "1")

    frame = load_sidecar_feature_frame(output_dir=tmp_path)
    assert list(frame.columns) == ["ds"]


def test_kalman_off_ignores_sidecar_columns(monkeypatch):
    ds = pd.date_range("2020-01-01", periods=30, freq="MS")
    actual = np.linspace(-30, 30, len(ds))
    base = pd.DataFrame(
        {
            "ds": ds,
            "actual": actual,
            "consensus_pred": actual + 1.0,
            "champion_pred": actual - 1.0,
        }
    )
    with_sidecar = base.assign(
        sidecar_accel__predicted_mom=actual + 100.0,
        sidecar_accel__confidence=1.0,
        sidecar_accel__predicted_accel_sign=1.0,
    )

    monkeypatch.setenv("NFP_SIDECAR_MODE", "off")
    plain, _ = kalman_fusion(base, base, trailing_window=6, use_model=True, use_nsa_accel=False)
    ignored, _ = kalman_fusion(with_sidecar, base, trailing_window=6, use_model=True, use_nsa_accel=False)

    pd.testing.assert_series_equal(plain["predicted"], ignored["predicted"], check_names=False)
    assert "sidecar_precision_share" not in ignored.columns


def test_kalman_sidecar_fusion_is_precision_capped(monkeypatch):
    ds = pd.date_range("2020-01-01", periods=30, freq="MS")
    actual = np.linspace(-30, 30, len(ds))
    overlap = pd.DataFrame(
        {
            "ds": ds,
            "actual": actual,
            "consensus_pred": actual + 1.0,
            "champion_pred": actual - 1.0,
            "sidecar_accel__predicted_mom": actual + 100.0,
            "sidecar_accel__confidence": 1.0,
        }
    )

    monkeypatch.setenv("NFP_SIDECAR_MODE", "fusion")
    monkeypatch.setenv("NFP_SIDECAR_MAX_PRECISION_SHARE", "0.15")
    fused, _ = kalman_fusion(overlap, overlap, trailing_window=6, use_model=True, use_nsa_accel=False)

    assert "sidecar_precision_share" in fused.columns
    assert fused["sidecar_precision_share"].max() <= 0.1500001
