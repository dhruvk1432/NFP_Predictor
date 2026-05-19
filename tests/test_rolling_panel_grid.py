import pandas as pd

from experiments.consensus_anchor_rolling_panel_grid import (
    GridSpec,
    _pool_prediction,
    _compute_track_record_pit,
    build_learned_router,
    build_panel_router_v2,
    build_rolling_panel,
)


def test_track_record_uses_only_operationally_available_actuals():
    panel = pd.DataFrame(
        [
            {
                "ds": pd.Timestamp("2024-01-01"),
                "ident": "A",
                "name": "A Econ",
                "forecast": 110.0,
                "first_release_date": pd.Timestamp("2024-02-01"),
            },
            {
                "ds": pd.Timestamp("2024-02-01"),
                "ident": "A",
                "name": "A Econ",
                "forecast": 500.0,
                "first_release_date": pd.Timestamp("2024-03-01"),
            },
            {
                "ds": pd.Timestamp("2024-01-01"),
                "ident": "B",
                "name": "B Econ",
                "forecast": 90.0,
                "first_release_date": pd.Timestamp("2024-02-01"),
            },
        ]
    )
    actuals = pd.DataFrame(
        [
            {
                "ds": pd.Timestamp("2024-01-01"),
                "actual": 100.0,
                "actual_available_date": pd.Timestamp("2024-02-02"),
            },
            {
                "ds": pd.Timestamp("2024-02-01"),
                "actual": 50.0,
                "actual_available_date": pd.Timestamp("2024-03-15"),
            },
        ]
    )

    track, trained_through, n_scorable = _compute_track_record_pit(
        panel,
        actuals,
        target_month=pd.Timestamp("2024-03-01"),
        cutoff=pd.Timestamp("2024-03-08"),
        track_window=2,
    )

    assert n_scorable == 1
    assert trained_through == pd.Timestamp("2024-01-01")
    assert set(track["ident"]) == {"A", "B"}
    assert track.set_index("ident").loc["A", "mae"] == 10.0
    assert track.set_index("ident").loc["A", "coverage"] == 1.0


def test_build_rolling_panel_filters_current_forecasts_by_release_cutoff():
    panel = pd.DataFrame(
        [
            {
                "ds": pd.Timestamp("2024-01-01"),
                "ident": "A",
                "name": "A Econ",
                "forecast": 105.0,
                "first_release_date": pd.Timestamp("2024-02-01"),
            },
            {
                "ds": pd.Timestamp("2024-01-01"),
                "ident": "B",
                "name": "B Econ",
                "forecast": 200.0,
                "first_release_date": pd.Timestamp("2024-02-01"),
            },
            {
                "ds": pd.Timestamp("2024-02-01"),
                "ident": "A",
                "name": "A Econ",
                "forecast": 125.0,
                "first_release_date": pd.Timestamp("2024-03-01"),
            },
            {
                "ds": pd.Timestamp("2024-02-01"),
                "ident": "B",
                "name": "B Econ",
                "forecast": 20.0,
                "first_release_date": pd.Timestamp("2024-03-10"),
            },
        ]
    )
    actuals = pd.DataFrame(
        [
            {
                "ds": pd.Timestamp("2024-01-01"),
                "actual": 100.0,
                "actual_available_date": pd.Timestamp("2024-02-02"),
            },
            {
                "ds": pd.Timestamp("2024-02-01"),
                "actual": 130.0,
                "actual_available_date": pd.Timestamp("2024-03-20"),
            },
        ]
    )
    release_map = pd.Series(
        [pd.Timestamp("2024-03-08")],
        index=[pd.Timestamp("2024-02-01")],
    )

    out = build_rolling_panel(
        GridSpec(track_window=1, top_n=2, min_coverage_pct=0.7),
        panel=panel,
        actuals=actuals,
        release_map=release_map,
        target_months=[pd.Timestamp("2024-02-01")],
    )

    assert out.loc[0, "panel_pred_raw"] == 125.0
    assert out.loc[0, "selected_names"] == "A Econ"
    assert out.loc[0, "latest_forecast_release"] < release_map.iloc[0]


def test_pooling_variants_are_deterministic():
    selected = pd.DataFrame(
        {
            "forecast": [100.0, 200.0, 300.0],
            "bias": [10.0, -10.0, 0.0],
            "mae": [10.0, 20.0, 40.0],
            "rmse": [12.0, 24.0, 48.0],
        }
    )

    assert _pool_prediction(selected, "equal_mean") == 200.0
    assert _pool_prediction(selected, "median") == 200.0
    assert _pool_prediction(selected, "bias_corrected_mean") == 200.0
    assert _pool_prediction(selected, "inv_mae_weighted_mean") < 150.0
    assert _pool_prediction(selected, "inv_rmse_weighted_mean") < 150.0


def test_router_v2_uses_only_prior_actuals_for_rule_choice():
    months = pd.date_range("2020-01-01", periods=30, freq="MS")
    actual = [100.0] * 24 + [300.0] * 6
    panel = [100.0] * 24 + [100.0] * 6
    kalman = [250.0] * 24 + [300.0] * 6
    base = pd.DataFrame(
        {
            "ds": months,
            "actual": actual,
            "consensus_pred": panel,
            "panel_consensus_mean": panel,
            "kalman_pred": kalman,
            "panel_size": 4,
            "panel_dispersion_std": 10.0,
            "selected_mean_coverage": 1.0,
            "selected_mean_mae": 1.0,
            "selected_mean_rmse": 1.0,
        }
    )

    out = build_panel_router_v2(
        base,
        panel_model_id="synthetic",
        min_history=24,
        objective="mae",
    )

    # Month 25 still chooses from months 1-24 only, where panel was perfect.
    assert out.loc[24, "selected_model"] == "panel"
    assert out.loc[24, "predicted"] == 100.0


def test_learned_router_training_excludes_current_month_actual():
    months = pd.date_range("2020-01-01", periods=30, freq="MS")
    actual = [100.0] * 24 + [300.0] * 6
    panel = [100.0] * 24 + [100.0] * 6
    kalman = [250.0] * 24 + [300.0] * 6
    base = pd.DataFrame(
        {
            "ds": months,
            "actual": actual,
            "consensus_pred": panel,
            "panel_consensus_mean": panel,
            "kalman_pred": kalman,
            "panel_size": 4,
            "panel_dispersion_std": 10.0,
            "selected_mean_coverage": 1.0,
            "selected_mean_mae": 1.0,
            "selected_mean_rmse": 1.0,
        }
    )

    out = build_learned_router(
        base,
        panel_model_id="synthetic",
        threshold=0.50,
        min_history=24,
    )

    # The first post-history month cannot use its own actual to discover the
    # regime break; it should still use the historically perfect panel.
    assert out.loc[24, "selected_model"] == "panel"
    assert out.loc[24, "predicted"] == 100.0
