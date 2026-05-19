import numpy as np
import pandas as pd


def test_walkforward_cv_score_small_window_does_not_score_full_window():
    from Train.Output_code.consensus_anchor_runner import _walkforward_cv_score

    n = 58
    overlap = pd.DataFrame({
        "ds": pd.date_range("2021-06-01", periods=n, freq="MS"),
        "actual": np.arange(n, dtype=float),
    })
    scored_ranges = []

    def fold_runner(full_overlap, eval_end):
        return pd.DataFrame({
            "ds": full_overlap.iloc[:eval_end]["ds"].values,
            "actual": full_overlap.iloc[:eval_end]["actual"].values,
            "predicted": full_overlap.iloc[:eval_end]["actual"].values,
        })

    def objective(actual, pred):
        scored_ranges.append(len(actual))
        return float(np.mean(np.abs(actual - pred)))

    score = _walkforward_cv_score(
        overlap,
        fold_runner,
        objective,
        n_splits=5,
        min_train=60,
    )

    assert score == 0.0
    assert scored_ranges
    assert sum(scored_ranges) < n
    assert max(scored_ranges) <= 3


def test_pit_adaptive_kalman_selects_params_from_prior_rows_only(monkeypatch):
    import Train.Output_code.consensus_anchor_runner as runner

    n = 35
    overlap = pd.DataFrame({
        "ds": pd.date_range("2022-01-01", periods=n, freq="MS"),
        "actual": np.zeros(n, dtype=float),
        "consensus_pred": np.zeros(n, dtype=float),
        "champion_pred": np.zeros(n, dtype=float),
    })

    def fake_kalman_fusion(overlap_df, consensus_df, trailing_window=18, **kwargs):
        pred = np.full(len(overlap_df), 100.0, dtype=float)
        if int(trailing_window) == 12:
            pred[:] = 0.0
            pred[30:] = 1000.0
        res = overlap_df[["ds", "actual", "consensus_pred"]].copy()
        res["predicted"] = pred
        res["error"] = res["actual"] - res["predicted"]
        metrics = runner.full_metrics(
            res["actual"].values,
            res["predicted"].values,
            "Kalman_Fusion",
            ds=res["ds"],
        )
        return res, metrics

    monkeypatch.setattr(runner, "kalman_fusion", fake_kalman_fusion)

    tuned, _, _ = runner.pit_adaptive_kalman_fusion(
        overlap,
        overlap,
        min_history=30,
        objective="mae",
    )

    row_31 = tuned.iloc[30]
    assert row_31["selection_history_n"] == 30
    assert row_31["selected_trailing_window"] == 12
    # If the current row's realized error leaked into selection, any non-12
    # candidate would be preferred because tw=12 intentionally fails at row 31.
    assert row_31["predicted"] == 1000.0


def test_pit_adaptive_kalman_selection_excludes_same_day_revised_actuals(monkeypatch):
    import Train.Output_code.consensus_anchor_runner as runner

    n = 35
    ds = pd.date_range("2022-01-01", periods=n, freq="MS")
    overlap = pd.DataFrame({
        "ds": ds,
        "actual": np.zeros(n, dtype=float),
        "target_release_date": ds + pd.DateOffset(months=1, days=4),
        "actual_available_date": ds + pd.DateOffset(months=2, days=4),
        "consensus_pred": np.zeros(n, dtype=float),
        "champion_pred": np.zeros(n, dtype=float),
    })

    def fake_kalman_fusion(overlap_df, consensus_df, trailing_window=18, **kwargs):
        res = overlap_df[[
            "ds", "actual", "target_release_date", "actual_available_date", "consensus_pred",
        ]].copy()
        res["predicted"] = 0.0
        res["error"] = 0.0
        metrics = runner.full_metrics(
            res["actual"].values,
            res["predicted"].values,
            "Kalman_Fusion",
            ds=res["ds"],
        )
        return res, metrics

    monkeypatch.setattr(runner, "kalman_fusion", fake_kalman_fusion)

    tuned, _, _ = runner.pit_adaptive_kalman_fusion(
        overlap,
        overlap,
        min_history=1,
        objective="mae",
    )

    # At row 31 (index 30), month 30's previous actual is released on the same
    # date as row 31's target release. Strict PIT excludes it, leaving 29
    # available historical actuals rather than 30 chronological prior rows.
    assert tuned.iloc[30]["selection_history_n"] == 29


def test_revised_training_label_mask_excludes_same_day_available_label():
    from Train.train_lightgbm_nfp import _available_label_mask_for_cutoff

    ds = pd.date_range("2020-01-01", periods=5, freq="MS")
    target = pd.DataFrame({
        "ds": ds,
        "y_mom": [10.0, 20.0, 30.0, 40.0, 50.0],
        "operational_available_date": [
            pd.Timestamp("2020-02-07"),
            pd.Timestamp("2020-03-06"),
            pd.Timestamp("2020-04-03"),
            pd.Timestamp("2020-05-08"),
            pd.Timestamp("2020-06-05"),
        ],
    })

    mask = _available_label_mask_for_cutoff(
        pd.Series(ds),
        target,
        pd.Timestamp("2020-05-08"),
        target_source="revised",
    )

    assert mask.tolist() == [True, True, True, False, False]
