import pandas as pd

from Train.Output_code.metrics import add_consensus_hit_rate_metrics


def test_consensus_hit_rate_excludes_only_covid_months():
    df = pd.DataFrame({
        "ds": pd.to_datetime([
            "2020-02-01",
            "2020-03-01",
            "2020-04-01",
            "2020-05-01",
            "2020-06-01",
        ]),
        "actual": [100.0, 100.0, 100.0, 100.0, 100.0],
        "predicted": [90.0, 100.0, 100.0, 100.0, 120.0],
        "consensus_pred": [80.0, 0.0, 0.0, 0.0, 110.0],
    })

    metrics = add_consensus_hit_rate_metrics({}, df)

    assert metrics["HitN_vs_ConsensusMean_NonCovid"] == 2
    assert metrics["HitWins_vs_ConsensusMean_NonCovid"] == 1
    assert metrics["HitLosses_vs_ConsensusMean_NonCovid"] == 1
    assert metrics["HitRate_vs_ConsensusMean_NonCovid"] == 0.5


def test_consensus_hit_rate_tracks_ties_separately():
    df = pd.DataFrame({
        "ds": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "actual": [100.0, 100.0],
        "predicted": [90.0, 105.0],
        "consensus_pred": [110.0, 95.0],
        "consensus_median_pred": [80.0, 103.0],
    })

    metrics = add_consensus_hit_rate_metrics({}, df)

    assert metrics["TieRate_vs_ConsensusMean"] == 1.0
    assert metrics["HitRate_vs_ConsensusMean"] == 0.0
    assert metrics["HitRate_vs_ConsensusMedian"] == 0.5
