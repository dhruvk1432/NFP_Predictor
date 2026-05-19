"""Unit tests for the economist-panel sidecar.

Asserts the PIT contract (``trained_through < ds`` for every output row),
backwards-compat of the per-step pool helper against synthetic panels, and
that the output frame carries every required sidecar schema column.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.sidecars.common import REQUIRED_PREDICTION_COLUMNS  # noqa: E402
from experiments.sidecars.economist_panel_sidecar import (  # noqa: E402
    PanelConfig,
    _compute_track_record,
    _latest_available_forecasts,
    _pool_step,
    _shape_for_sidecar,
    _trained_through_for_step,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_panel():
    """4 economists with monthly forecasts spanning 2010-2023.

    Each economist has a known persistent bias and noise:
      ECON_A: bias +50,  std 30  (small, persistent over-forecaster)
      ECON_B: bias  -30, std 40
      ECON_C: bias   +5, std 50  (most accurate on average)
      ECON_D: bias  +80, std 80  (large noisy bias)
    Actuals follow N(150, 50).
    """
    rng = np.random.default_rng(0)
    ds = pd.date_range("2010-01-01", "2023-12-01", freq="MS")
    actuals = pd.Series(
        rng.normal(150, 50, len(ds)).astype(float),
        index=ds,
        name="actual",
    )

    rows = []
    profiles = [
        ("US_A", "ECON_A", 50.0, 30.0),
        ("US_B", "ECON_B", -30.0, 40.0),
        ("US_C", "ECON_C", 5.0, 50.0),
        ("US_D", "ECON_D", 80.0, 80.0),
    ]
    for ident, name, bias, std in profiles:
        noise = rng.normal(0, std, len(ds))
        forecast = actuals.values + bias + noise
        for i, d in enumerate(ds):
            rows.append({
                "ds": d,
                "ident": ident,
                "name": name,
                "forecast": float(forecast[i]),
                # Forecasts are filed strictly before the target month.
                "first_release_date": d - pd.DateOffset(days=2),
            })
    return pd.DataFrame(rows), actuals


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_track_record_pit_strict(synthetic_panel):
    """Track record at target month M only sees data with first_release_date < cutoff
    AND with actuals from ds < M."""
    panel, actuals = synthetic_panel
    target = pd.Timestamp("2022-06-01")
    cutoff = target + pd.DateOffset(days=5)  # NFP release of M
    track = _compute_track_record(
        panel=panel,
        actuals=actuals,
        target_month=target,
        cutoff=cutoff,
        track_window=24,
    )
    assert not track.empty
    # Every economist should have ≤ track_window observations
    assert (track["n"] <= 24).all()
    # The known-bias economist A must come out with ≈ +50 bias
    bias_a = float(track[track["ident"] == "US_A"]["bias"].iloc[0])
    assert 40 < bias_a < 60, f"ECON_A trailing bias {bias_a} should be ~50"


def test_latest_available_forecasts_uses_last_submission_before_cutoff():
    panel = pd.DataFrame([
        {
            "ds": pd.Timestamp("2022-05-01"),
            "ident": "US_A",
            "name": "ECON_A",
            "forecast": 100.0,
            "first_release_date": pd.Timestamp("2022-05-20"),
        },
        {
            "ds": pd.Timestamp("2022-05-01"),
            "ident": "US_A",
            "name": "ECON_A",
            "forecast": 130.0,
            "first_release_date": pd.Timestamp("2022-06-01"),
        },
        {
            "ds": pd.Timestamp("2022-05-01"),
            "ident": "US_A",
            "name": "ECON_A",
            "forecast": 999.0,
            "first_release_date": pd.Timestamp("2022-06-10"),
        },
    ])
    cutoff = pd.Timestamp("2022-06-03")

    latest = _latest_available_forecasts(
        panel[panel["first_release_date"] < cutoff]
    )

    assert len(latest) == 1
    assert float(latest["forecast"].iloc[0]) == 130.0


def test_track_record_scores_last_available_submission():
    actuals = pd.Series(
        [100.0],
        index=[pd.Timestamp("2022-05-01")],
        name="actual",
    )
    panel = pd.DataFrame([
        {
            "ds": pd.Timestamp("2022-05-01"),
            "ident": "US_A",
            "name": "ECON_A",
            "forecast": 100.0,
            "first_release_date": pd.Timestamp("2022-05-20"),
        },
        {
            "ds": pd.Timestamp("2022-05-01"),
            "ident": "US_A",
            "name": "ECON_A",
            "forecast": 130.0,
            "first_release_date": pd.Timestamp("2022-06-01"),
        },
        {
            "ds": pd.Timestamp("2022-05-01"),
            "ident": "US_A",
            "name": "ECON_A",
            "forecast": 999.0,
            "first_release_date": pd.Timestamp("2022-06-10"),
        },
    ])

    track = _compute_track_record(
        panel=panel,
        actuals=actuals,
        target_month=pd.Timestamp("2022-06-01"),
        cutoff=pd.Timestamp("2022-06-03"),
        track_window=1,
    )

    assert len(track) == 1
    assert float(track["bias"].iloc[0]) == pytest.approx(30.0)
    assert float(track["mae"].iloc[0]) == pytest.approx(30.0)


def test_track_record_excludes_actuals_not_operationally_available(synthetic_panel):
    panel, actuals = synthetic_panel
    target = pd.Timestamp("2022-06-01")
    cutoff = target + pd.DateOffset(days=5)
    window_month = target - pd.DateOffset(months=1)
    actuals = actuals.copy()
    actuals.attrs["actual_available_date_by_ds"] = pd.Series(
        [cutoff],
        index=[window_month],
        name="actual_available_date",
    )

    track = _compute_track_record(
        panel=panel,
        actuals=actuals,
        target_month=target,
        cutoff=cutoff,
        track_window=1,
    )

    assert track.empty


def test_pool_step_emits_all_variants(synthetic_panel):
    """_pool_step always emits all candidate keys (NaN if not computable)."""
    panel, actuals = synthetic_panel
    target = pd.Timestamp("2022-06-01")
    cutoff = target + pd.DateOffset(days=5)
    eligible = panel[
        (panel["ds"] == target) & (panel["first_release_date"] < cutoff)
    ][["ident", "name", "forecast"]]
    track = _compute_track_record(panel, actuals, target, cutoff, 24)

    cfg = PanelConfig(top_n_variants=(3,), primary_top_n=3, min_coverage_pct=0.166)
    out = _pool_step(eligible, track, cfg)

    required_keys = {
        "predicted_mom",
        "predicted_mom_robust_median",
        "predicted_mom_trimmed10",
        "predicted_mom_top3_simple",
        "predicted_mom_top3_bc_simple",
        "panel_n",
        "panel_n_calibrated",
        "panel_top3_size",
        "panel_top3_mean_mae",
        "panel_top3_mean_rmse",
    }
    assert required_keys.issubset(set(out.keys())), (
        f"Missing keys: {required_keys - set(out.keys())}"
    )
    # The 4-economist fixture has all calibrated at this date.
    assert out["panel_n_calibrated"] == 4
    # top3_simple should be a finite number.
    assert np.isfinite(out["predicted_mom_top3_simple"])
    # The primary channel must equal the top-3 simple variant when
    # apply_bias_correction is False (the test default).
    assert out["predicted_mom"] == pytest.approx(out["predicted_mom_top3_simple"])


def test_pool_step_selects_lowest_mae_economist(synthetic_panel):
    """With top_n=1, the auto-selection should pick the economist with the
    lowest trailing MAE — by construction that's ECON_C in our fixture (bias
    +5, std 50 vs others with larger bias)."""
    panel, actuals = synthetic_panel
    target = pd.Timestamp("2022-06-01")
    cutoff = target + pd.DateOffset(days=5)
    eligible = panel[
        (panel["ds"] == target) & (panel["first_release_date"] < cutoff)
    ][["ident", "name", "forecast"]]
    track = _compute_track_record(panel, actuals, target, cutoff, 24)

    cfg = PanelConfig(top_n_variants=(1,), primary_top_n=1, min_coverage_pct=0.166,
                      apply_bias_correction=False)
    out = _pool_step(eligible, track, cfg)
    # The top-1 simple equals ECON_C's forecast at this month.
    econ_c_forecast = float(
        eligible[eligible["ident"] == "US_C"]["forecast"].iloc[0]
    )
    assert out["predicted_mom_top1_simple"] == pytest.approx(econ_c_forecast)
    assert out["predicted_mom"] == pytest.approx(econ_c_forecast)


def test_shape_for_sidecar_emits_required_columns(synthetic_panel):
    """After ``_shape_for_sidecar`` + ``standardize_predictions`` (the path
    write_sidecar_artifacts takes), the frame must satisfy the sidecar
    schema contract on every required column."""
    from experiments.sidecars.common import standardize_predictions

    panel, actuals = synthetic_panel
    cfg = PanelConfig(top_n_variants=(3,), primary_top_n=3, min_coverage_pct=0.166)
    target = pd.Timestamp("2022-06-01")
    cutoff = target + pd.DateOffset(days=5)
    eligible = panel[
        (panel["ds"] == target) & (panel["first_release_date"] < cutoff)
    ][["ident", "name", "forecast"]]
    track = _compute_track_record(panel, actuals, target, cutoff, 24)
    stats = _pool_step(eligible, track, cfg)

    trained = _trained_through_for_step(panel, actuals, target, cutoff)
    row = {"ds": target, "actual_mom": float(actuals[target]),
           "trained_through": trained, **stats}
    shaped = _shape_for_sidecar(pd.DataFrame([row]))
    finalized = standardize_predictions(
        shaped, model_id="economist_panel_test", target_space=cfg.target_type,
    )

    missing = REQUIRED_PREDICTION_COLUMNS - set(finalized.columns)
    assert not missing, f"Missing required cols after finalization: {missing}"
    # PIT contract preserved through standardize_predictions
    assert (finalized["trained_through"] < finalized["ds"]).all()


def test_pit_invariant_holds_for_every_row(synthetic_panel):
    """trained_through must be strictly before ds for every row."""
    panel, actuals = synthetic_panel
    cfg = PanelConfig(top_n_variants=(3,), primary_top_n=3, min_coverage_pct=0.166)
    target_months = pd.date_range("2018-01-01", "2023-06-01", freq="MS")
    rows = []
    for target in target_months:
        cutoff = target + pd.DateOffset(days=5)
        eligible = panel[
            (panel["ds"] == target) & (panel["first_release_date"] < cutoff)
        ][["ident", "name", "forecast"]]
        track = _compute_track_record(panel, actuals, target, cutoff, 24)
        stats = _pool_step(eligible, track, cfg)
        trained = _trained_through_for_step(panel, actuals, target, cutoff)
        rows.append({
            "ds": target,
            "actual_mom": float(actuals[target]),
            "trained_through": trained,
            **stats,
        })
    shaped = _shape_for_sidecar(pd.DataFrame(rows))
    # Strict PIT contract.
    assert (shaped["trained_through"] < shaped["ds"]).all(), \
        "trained_through must be strictly before ds for every row"


def test_no_calibrated_fallback_returns_robust_median(synthetic_panel):
    """When no economist has enough prior history, predicted_mom should
    fall back to the robust median of eligible forecasts."""
    panel, actuals = synthetic_panel
    cfg = PanelConfig(
        top_n_variants=(3,),
        primary_top_n=3,
        # Require an absurd coverage threshold so nobody calibrates.
        min_coverage_pct=2.0,
    )
    target = pd.Timestamp("2022-06-01")
    cutoff = target + pd.DateOffset(days=5)
    eligible = panel[
        (panel["ds"] == target) & (panel["first_release_date"] < cutoff)
    ][["ident", "name", "forecast"]]
    track = _compute_track_record(panel, actuals, target, cutoff, 24)
    out = _pool_step(eligible, track, cfg)
    expected_median = float(eligible["forecast"].median())
    assert out["predicted_mom"] == pytest.approx(expected_median)
    assert out["panel_n_calibrated"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
