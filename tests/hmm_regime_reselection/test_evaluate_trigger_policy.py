import pandas as pd

from experiments.hmm_regime_reselection_study.evaluate_trigger_policy import evaluate


def test_evaluate_trigger_policy_reports_precision_recall_without_training():
    rows = [
        {
            "step_date": "2008-01",
            "final_reselect": True,
            "hmm_regime_label": "volatile_down",
            "hmm_trigger_class": "regime_shift",
            "event_window": "global_financial_crisis",
            "hmm_surprise_ratio": 2.2,
        },
        {
            "step_date": "2008-02",
            "final_reselect": False,
            "hmm_regime_label": "volatile_down",
            "hmm_trigger_class": "no_shift",
            "event_window": "global_financial_crisis",
            "hmm_surprise_ratio": 1.2,
        },
        {
            "step_date": "2020-03",
            "final_reselect": True,
            "hmm_regime_label": "crash",
            "hmm_trigger_class": "surprise",
            "event_window": "covid_crash",
            "hmm_surprise_ratio": 3.0,
        },
        {
            "step_date": "2021-01",
            "final_reselect": True,
            "hmm_regime_label": "crash",
            "hmm_trigger_class": "regime_shift",
            "event_window": None,
            "hmm_surprise_ratio": 1.1,
        },
    ]
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["step_date"]).dt.to_period("M").dt.to_timestamp()
    df["hmm_month_of_year"] = df["ds"].dt.month

    result = evaluate(df, cluster_gap_months=6, last_n_months=60)

    assert result["summary"]["n_triggers"] == 3
    assert result["summary"]["jan_jul_triggers"] == 2
    assert result["summary"]["cluster_event_precision"] == 2 / 3
    assert result["event_recall"].set_index("event_window").loc["global_financial_crisis", "hit"]
