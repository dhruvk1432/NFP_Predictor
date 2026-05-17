import numpy as np
import pandas as pd


def test_economist_panel_pit_loader_builds_cross_sectional_anchor(tmp_path, monkeypatch):
    from Train.Output_code import consensus_anchor_runner as car

    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()

    cols = car.ECONOMIST_PANEL_FORECAST_COLS
    jan = pd.DataFrame({
        "date": [pd.Timestamp("2021-01-01")],
        cols[0]: [100.0],
        cols[1]: [110.0],
        cols[2]: [90.0],
        cols[3]: [np.nan],
        cols[4]: [120.0],
    })
    feb = pd.DataFrame({
        "date": [pd.Timestamp("2021-02-01")],
        cols[0]: [200.0],
        cols[1]: [np.nan],
        cols[2]: [np.nan],
        cols[3]: [220.0],
        cols[4]: [210.0],
    })
    jan.to_parquet(snapshot_dir / "2021-01.parquet")
    feb.to_parquet(snapshot_dir / "2021-02.parquet")

    monkeypatch.setattr(car, "get_master_snapshots_dir", lambda *_: snapshot_dir)

    out = car._load_economist_panel_pit("sa", "revised")

    assert list(out["ds"]) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01")]
    assert out.loc[0, "panel_consensus_count"] == 4
    assert out.loc[0, "panel_consensus_mean"] == 105.0
    assert out.loc[0, "panel_consensus_median"] == 105.0
    assert out.loc[1, "panel_consensus_count"] == 3
    assert out.loc[1, "panel_consensus_mean"] == 210.0
    assert out.loc[1, "panel_NFP_Forecast_AIB"] == 200.0
