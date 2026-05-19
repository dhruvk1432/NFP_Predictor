import numpy as np
import pandas as pd


def test_panel_loader_uses_dynamic_panel_aggregates_when_fixed_list_empty(tmp_path, monkeypatch):
    from Train.Output_code import consensus_anchor_runner as car

    snap_dir = tmp_path / "master_snapshots" / "sa" / "revised" / "decades" / "2020s" / "2025"
    snap_dir.mkdir(parents=True)
    pd.DataFrame({
        "date": [pd.Timestamp("2025-01-01")],
        "NFP_Forecast_Dynamic_Top10_k12": [175.0],
        "NFP_Forecast_Dynamic_Top4_k12": [170.0],
        "NFP_Forecast_Dynamic_Top15_k12": [180.0],
        "NFP_Forecast_Dynamic_RobustMedian": [172.0],
        "NFP_Forecast_Dynamic_TrimmedMean10": [174.0],
        "NFP_Forecast_Dynamic_PanelN": [25.0],
        "NFP_Forecast_Dynamic_NCalibrated": [10.0],
        "NFP_Forecast_Dynamic_DispersionStd": [18.0],
        "NFP_Forecast_Dynamic_DispersionIqr": [24.0],
        "NFP_Forecast_Dynamic_Top10TrackMae": [63.0],
    }).to_parquet(snap_dir / "2025-01.parquet", index=False)

    monkeypatch.setattr(car, "ECONOMIST_PANEL_FORECAST_COLS", [])
    monkeypatch.setattr(car, "get_master_snapshots_dir", lambda *_args, **_kwargs: tmp_path / "master_snapshots" / "sa" / "revised" / "decades")

    out = car._load_economist_panel_pit()

    assert len(out) == 1
    row = out.iloc[0]
    assert row["panel_consensus_mean"] == 175.0
    assert row["panel_consensus_median"] == 172.0
    assert row["panel_consensus_count"] == 10
    assert row["panel_consensus_std"] == 18.0
    assert row["panel_source"] == "dynamic_panel_aggregates"


def test_panel_kalman_router_uses_only_prior_actuals_for_rule_choice():
    from Train.Output_code import consensus_anchor_runner as car

    ds = pd.date_range("2021-01-01", periods=6, freq="MS")
    actual = [100.0] * 6
    panel_pred = [100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0]
    kalman_pred = [1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0]

    panel_results = pd.DataFrame({
        "ds": ds,
        "actual": actual,
        "predicted": panel_pred,
        "panel_consensus_mean": panel_pred,
        "consensus_pred": [100.0] * 6,
    })
    kalman_df = pd.DataFrame({
        "ds": ds,
        "actual": actual,
        "predicted": kalman_pred,
    })

    out, _ = car.build_panel_kalman_router(
        panel_results,
        kalman_df,
        min_history=2,
        objective="mae",
    )

    # Through 2021-03 the panel has been perfect and Kalman has been bad, so
    # the 2021-04 row must still pick panel even though future rows would prove
    # Kalman better. A future-looking router would pick Kalman here.
    apr = out[out["ds"] == pd.Timestamp("2021-04-01")].iloc[0]
    assert apr["selected_rule"] == "panel"
    assert apr["predicted"] == 1000.0


def test_kalman_fusion_excludes_same_day_prior_revised_actual():
    from Train.Output_code import consensus_anchor_runner as car

    ds = pd.date_range("2021-01-01", periods=6, freq="MS")
    actual = np.arange(6, dtype=float) * 10.0
    release_dates = ds + pd.DateOffset(months=1, days=4)
    actual_available = ds + pd.DateOffset(months=2, days=4)
    overlap = pd.DataFrame({
        "ds": ds,
        "actual": actual,
        "target_release_date": release_dates,
        "actual_available_date": actual_available,
        "consensus_pred": actual + 1.0,
        "champion_pred": actual + 2.0,
    })

    out, _ = car.kalman_fusion(
        overlap,
        overlap,
        use_nsa_accel=False,
    )

    feb = out[out["ds"] == pd.Timestamp("2021-02-01")].iloc[0]
    mar = out[out["ds"] == pd.Timestamp("2021-03-01")].iloc[0]
    assert feb["history_available_n"] == 0
    assert pd.isna(feb["latest_available_actual_ds"])
    assert mar["history_available_n"] == 1
    assert mar["latest_available_actual_ds"] == pd.Timestamp("2021-01-01")


def test_panel_kalman_router_excludes_same_day_prior_revised_actuals():
    from Train.Output_code import consensus_anchor_runner as car

    ds = pd.date_range("2021-01-01", periods=6, freq="MS")
    actual = [100.0] * 6
    release_dates = ds + pd.DateOffset(months=1, days=4)
    actual_available = ds + pd.DateOffset(months=2, days=4)
    panel_pred = [100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0]
    kalman_pred = [1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0]

    panel_results = pd.DataFrame({
        "ds": ds,
        "actual": actual,
        "target_release_date": release_dates,
        "actual_available_date": actual_available,
        "predicted": panel_pred,
        "panel_consensus_mean": panel_pred,
        "consensus_pred": [100.0] * 6,
    })
    kalman_df = pd.DataFrame({
        "ds": ds,
        "actual": actual,
        "predicted": kalman_pred,
    })

    out, _ = car.build_panel_kalman_router(
        panel_results,
        kalman_df,
        min_history=2,
        objective="mae",
    )

    apr = out[out["ds"] == pd.Timestamp("2021-04-01")].iloc[0]
    assert apr["router_history_available_n"] == 2
    assert apr["router_latest_available_actual_ds"] == pd.Timestamp("2021-02-01")


def test_panel_kalman_router_can_switch_to_kalman_with_live_panel_from_recent_pit_history():
    from Train.Output_code import consensus_anchor_runner as car

    ds = pd.date_range("2020-01-01", periods=40, freq="MS")
    actual = np.full(len(ds), 100.0)
    panel_pred = np.full(len(ds), 100.0)
    kalman_pred = np.full(len(ds), 300.0)

    # The older prefix favors panel; the latest twelve operationally available
    # months favor Kalman. The target row still has a live panel value, so this
    # specifically tests live-panel routing rather than missing-panel fallback.
    panel_pred[24:] = 220.0
    kalman_pred[24:] = 100.0

    panel_results = pd.DataFrame({
        "ds": ds,
        "actual": actual,
        "target_release_date": ds + pd.DateOffset(months=1, days=4),
        "actual_available_date": ds + pd.DateOffset(months=1, days=2),
        "predicted": panel_pred,
        "panel_consensus_mean": panel_pred,
        "panel_consensus_count": 5,
        "panel_consensus_std": 12.0,
        "consensus_pred": panel_pred,
    })
    kalman_df = pd.DataFrame({
        "ds": ds,
        "actual": actual,
        "predicted": kalman_pred,
    })

    out, metrics = car.build_panel_kalman_router(
        panel_results,
        kalman_df,
        min_history=12,
        objective="mae",
        selection_lookback=12,
    )

    jan_2022 = out[out["ds"] == pd.Timestamp("2022-01-01")].iloc[0]
    jan_2023 = out[out["ds"] == pd.Timestamp("2023-01-01")].iloc[0]

    assert jan_2022["selected_source"] == "panel"
    assert jan_2023["selected_source"] == "kalman"
    assert jan_2023["panel_consensus_mean"] == 220.0
    assert jan_2023["predicted"] == 100.0
    assert metrics["Router_Live_Panel_Kalman_Count"] > 0


def test_panel_router_mae_hit_penalizes_losing_to_consensus():
    from Train.Output_code import consensus_anchor_runner as car

    actual = np.array([100.0, 100.0, 100.0])
    consensus = np.array([100.0, 100.0, 100.0])
    pred = np.array([150.0, 150.0, 150.0])
    ds = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]).to_numpy()

    mae_score = car._panel_router_score(actual, pred, "mae")
    mae_hit_score = car._panel_router_score(
        actual,
        pred,
        "mae_hit",
        ds=ds,
        consensus=consensus,
    )

    assert mae_hit_score > mae_score


def test_sa_challenger_kalman_excludes_same_day_prior_revised_actual(monkeypatch):
    from Train.Output_code import sa_consensus_anchor_runner as sa_car

    ds = pd.date_range("2021-01-01", periods=6, freq="MS")
    actual = np.arange(6, dtype=float) * 10.0
    frame = pd.DataFrame({
        "ds": ds,
        "actual_mom": actual,
        "target_release_date": ds + pd.DateOffset(months=1, days=4),
        "actual_available_date": ds + pd.DateOffset(months=2, days=4),
        "consensus_pred": actual + 1.0,
    })

    def fake_build_fusion_frame(cfg):
        return frame.copy(), frame[["ds", "consensus_pred"]].copy(), ["consensus"]

    monkeypatch.setattr(sa_car, "build_fusion_frame", fake_build_fusion_frame)

    out, _, _ = sa_car.run_sa_kalman_fusion(sa_car.FusionConfig())

    feb = out[out["ds"] == pd.Timestamp("2021-02-01")].iloc[0]
    mar = out[out["ds"] == pd.Timestamp("2021-03-01")].iloc[0]
    assert feb["history_available_n"] == 0
    assert pd.isna(feb["latest_available_actual_ds"])
    assert mar["history_available_n"] == 1
    assert mar["latest_available_actual_ds"] == pd.Timestamp("2021-01-01")


def test_panel_replacement_builder_is_pit_clean():
    from Train.Output_code import consensus_anchor_runner as car

    panel = pd.DataFrame(
        [
            {
                "ds": pd.Timestamp("2021-01-01"),
                "ident": "A",
                "name": "A Econ",
                "forecast": 105.0,
                "first_release_date": pd.Timestamp("2021-02-01"),
            },
            {
                "ds": pd.Timestamp("2021-01-01"),
                "ident": "B",
                "name": "B Econ",
                "forecast": 150.0,
                "first_release_date": pd.Timestamp("2021-02-01"),
            },
            {
                "ds": pd.Timestamp("2021-02-01"),
                "ident": "A",
                "name": "A Econ",
                "forecast": 125.0,
                "first_release_date": pd.Timestamp("2021-03-01"),
            },
            {
                "ds": pd.Timestamp("2021-02-01"),
                "ident": "B",
                "name": "B Econ",
                "forecast": 10.0,
                "first_release_date": pd.Timestamp("2021-03-10"),
            },
        ]
    )
    actuals = pd.DataFrame(
        [
            {
                "ds": pd.Timestamp("2021-01-01"),
                "actual": 100.0,
                "actual_available_date": pd.Timestamp("2021-02-05"),
            },
            {
                "ds": pd.Timestamp("2021-02-01"),
                "actual": 130.0,
                "actual_available_date": pd.Timestamp("2021-03-15"),
            },
        ]
    )
    release_map = pd.Series(
        [pd.Timestamp("2021-03-05")],
        index=[pd.Timestamp("2021-02-01")],
    )

    out = car._build_rolling_panel_replacement(
        panel=panel,
        actuals=actuals,
        release_map=release_map,
        target_months=[pd.Timestamp("2021-02-01")],
        config={
            "track_window": 1,
            "top_n": 2,
            "min_coverage_pct": 0.7,
            "pooling": "median",
            "skip_covid_track_record": False,
        },
    )

    row = out.iloc[0]
    assert row["panel_replacement_pred"] == 125.0
    assert row["panel_replacement_selected_names"] == "A Econ"
    assert row["panel_replacement_trained_through"] < row["ds"]
    assert row["panel_replacement_latest_forecast_release"] < row["panel_replacement_target_release_date"]


def test_panel_replacement_selection_is_dynamic_by_month():
    from Train.Output_code import consensus_anchor_runner as car

    panel = pd.DataFrame(
        [
            {
                "ds": pd.Timestamp("2021-01-01"),
                "ident": "A",
                "name": "A Econ",
                "forecast": 100.0,
                "first_release_date": pd.Timestamp("2021-02-01"),
            },
            {
                "ds": pd.Timestamp("2021-01-01"),
                "ident": "B",
                "name": "B Econ",
                "forecast": 300.0,
                "first_release_date": pd.Timestamp("2021-02-01"),
            },
            {
                "ds": pd.Timestamp("2021-02-01"),
                "ident": "A",
                "name": "A Econ",
                "forecast": 300.0,
                "first_release_date": pd.Timestamp("2021-03-01"),
            },
            {
                "ds": pd.Timestamp("2021-02-01"),
                "ident": "B",
                "name": "B Econ",
                "forecast": 120.0,
                "first_release_date": pd.Timestamp("2021-03-01"),
            },
            {
                "ds": pd.Timestamp("2021-03-01"),
                "ident": "A",
                "name": "A Econ",
                "forecast": 50.0,
                "first_release_date": pd.Timestamp("2021-04-01"),
            },
            {
                "ds": pd.Timestamp("2021-03-01"),
                "ident": "B",
                "name": "B Econ",
                "forecast": 60.0,
                "first_release_date": pd.Timestamp("2021-04-01"),
            },
        ]
    )
    actuals = pd.DataFrame(
        [
            {
                "ds": pd.Timestamp("2021-01-01"),
                "actual": 100.0,
                "actual_available_date": pd.Timestamp("2021-02-05"),
            },
            {
                "ds": pd.Timestamp("2021-02-01"),
                "actual": 120.0,
                "actual_available_date": pd.Timestamp("2021-03-10"),
            },
        ]
    )
    release_map = pd.Series(
        [pd.Timestamp("2021-03-05"), pd.Timestamp("2021-04-05")],
        index=[pd.Timestamp("2021-02-01"), pd.Timestamp("2021-03-01")],
    )

    out = car._build_rolling_panel_replacement(
        panel=panel,
        actuals=actuals,
        release_map=release_map,
        target_months=[pd.Timestamp("2021-02-01"), pd.Timestamp("2021-03-01")],
        config={
            "track_window": 1,
            "top_n": 1,
            "min_coverage_pct": 1.0,
            "pooling": "median",
            "skip_covid_track_record": False,
        },
    )

    selected = dict(zip(out["ds"], out["panel_replacement_selected_names"]))
    assert selected[pd.Timestamp("2021-02-01")] == "A Econ"
    assert selected[pd.Timestamp("2021-03-01")] == "B Econ"


def test_panel_replaces_consensus_kalman_falls_back_to_consensus():
    from Train.Output_code import consensus_anchor_runner as car

    months = pd.date_range("2020-01-01", periods=30, freq="MS")
    actual = np.linspace(100.0, 129.0, len(months))
    overlap = pd.DataFrame(
        {
            "ds": months,
            "actual": actual,
            "consensus_pred": actual + 10.0,
            "champion_pred": actual + 20.0,
            "nsa_pred": actual + 15.0,
            "panel_replacement_pred": [np.nan] * 29 + [90.0],
        }
    )
    consensus_df = overlap[["ds", "actual", "consensus_pred"]].copy()

    out, metrics, manifest = car.build_panel_replaces_consensus_kalman(
        overlap,
        consensus_df,
        config={
            "trailing_window": 18,
            "nsa_weight_scale": 0.40,
            "track_window": 8,
            "top_n": 8,
            "min_coverage_pct": 0.80,
            "pooling": "median",
        },
    )

    assert metrics["Forecast"] == "Panel_Replaces_Consensus_Kalman"
    assert out.loc[0, "anchor_source"] == "consensus_fallback"
    assert out.loc[29, "anchor_source"] == "rolling_panel"
    assert manifest["anchor_source_counts"]["consensus_fallback"] == 29


def test_train_all_validation_requires_panel_kalman_router(tmp_path):
    from Train.train_lightgbm_nfp import validate_post_train_all_artifacts

    model_save_dir = tmp_path / "models" / "lightgbm_nfp"
    metrics_dir = tmp_path / "backtest"
    output_root = tmp_path / "_output"
    model_id = "nsa_first_revised"

    def touch(path) -> None:
        p = tmp_path / path if isinstance(path, str) else path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ok")

    touch(model_save_dir / model_id / f"lightgbm_{model_id}_model.txt")
    touch(model_save_dir / model_id / f"lightgbm_{model_id}_metadata.pkl")
    touch(metrics_dir / f"{model_id}_metrics.json")
    touch(model_save_dir / "model_comparison.csv")
    touch(model_save_dir / "model_comparison.html")
    touch(output_root / "NSA_prediction" / "backtest_results.csv")
    touch(output_root / "NSA_prediction" / "summary_statistics.csv")
    touch(output_root / "NSA_prediction" / "feature_importance.csv")
    touch(output_root / "NSA_plus_adjustment" / "backtest_results.csv")
    touch(output_root / "NSA_plus_adjustment" / "summary_statistics.csv")
    touch(output_root / "consensus_anchor" / "main_models.json")
    touch(output_root / "consensus_anchor" / "baseline_consensus" / "backtest_results.csv")
    touch(output_root / "consensus_anchor" / "kalman_fusion" / "backtest_results.csv")
    touch(output_root / "consensus_anchor" / "kalman_fusion" / "summary_statistics.csv")
    touch(output_root / "consensus_anchor" / "comparison_metrics.csv")
    touch(output_root / "consensus_anchor" / "comparison_overlay.png")
    touch(output_root / "consensus_anchor" / "comparison_metrics.png")
    touch(output_root / "consensus_anchor" / "comparison_scorecard.html")

    validation = validate_post_train_all_artifacts(
        run_model_ids=[model_id],
        model_save_dir=model_save_dir,
        metrics_dir=metrics_dir,
        output_root=output_root,
    )

    assert not validation["ok"]
    assert any("panel_kalman_router" in item for item in validation["missing_required"])


def test_train_all_validation_requires_panel_replacement_when_flagged(tmp_path, monkeypatch):
    from Train.train_lightgbm_nfp import validate_post_train_all_artifacts

    monkeypatch.setenv("NFP_ENABLE_PANEL_REPLACES_CONSENSUS_KALMAN", "1")

    model_save_dir = tmp_path / "models" / "lightgbm_nfp"
    metrics_dir = tmp_path / "backtest"
    output_root = tmp_path / "_output"
    model_id = "nsa_first_revised"

    def touch(path) -> None:
        p = tmp_path / path if isinstance(path, str) else path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ok")

    touch(model_save_dir / model_id / f"lightgbm_{model_id}_model.txt")
    touch(model_save_dir / model_id / f"lightgbm_{model_id}_metadata.pkl")
    touch(metrics_dir / f"{model_id}_metrics.json")
    touch(model_save_dir / "model_comparison.csv")
    touch(model_save_dir / "model_comparison.html")
    touch(output_root / "NSA_prediction" / "backtest_results.csv")
    touch(output_root / "NSA_prediction" / "summary_statistics.csv")
    touch(output_root / "NSA_prediction" / "feature_importance.csv")
    touch(output_root / "NSA_plus_adjustment" / "backtest_results.csv")
    touch(output_root / "NSA_plus_adjustment" / "summary_statistics.csv")
    touch(output_root / "consensus_anchor" / "main_models.json")
    touch(output_root / "consensus_anchor" / "baseline_consensus" / "backtest_results.csv")
    touch(output_root / "consensus_anchor" / "kalman_fusion" / "backtest_results.csv")
    touch(output_root / "consensus_anchor" / "kalman_fusion" / "summary_statistics.csv")
    touch(output_root / "consensus_anchor" / "panel_kalman_router" / "backtest_results.csv")
    touch(output_root / "consensus_anchor" / "panel_kalman_router" / "summary_statistics.csv")
    touch(output_root / "consensus_anchor" / "panel_kalman_router" / "router_manifest.json")
    touch(output_root / "consensus_anchor" / "comparison_metrics.csv")
    touch(output_root / "consensus_anchor" / "comparison_overlay.png")
    touch(output_root / "consensus_anchor" / "comparison_metrics.png")
    touch(output_root / "consensus_anchor" / "comparison_scorecard.html")

    validation = validate_post_train_all_artifacts(
        run_model_ids=[model_id],
        model_save_dir=model_save_dir,
        metrics_dir=metrics_dir,
        output_root=output_root,
    )

    assert not validation["ok"]
    assert any("panel_replaces_consensus_kalman" in item for item in validation["missing_required"])
