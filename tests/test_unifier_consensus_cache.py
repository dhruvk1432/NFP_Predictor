import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from Data_ETA_Pipeline.load_unifier_data import (  # noqa: E402
    CONSENSUS_SERIES,
    _missing_consensus_series_in_latest_snapshot,
)
from Data_ETA_Pipeline.utils import get_snapshot_path  # noqa: E402


def _write_unifier_snapshot(base_dir: Path, obs_month: str, series_names: list[str]) -> None:
    path = get_snapshot_path(base_dir, pd.Timestamp(obs_month))
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "date": pd.Timestamp("2026-01-01"),
            "release_date": pd.Timestamp("2026-01-31"),
            "snapshot_date": pd.Timestamp(obs_month),
            "series_name": name,
            "series_code": name,
            "value": 1.0,
        }
        for name in series_names
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_existing_unifier_cache_detects_missing_consensus_median(tmp_path):
    release_map = {
        pd.Timestamp("2026-04-01"): pd.Timestamp("2026-05-01"),
    }
    _write_unifier_snapshot(
        tmp_path,
        "2026-04-01",
        ["NFP_Consensus_Mean", "NFP_Consensus_Mean_lag_1m"],
    )

    missing = _missing_consensus_series_in_latest_snapshot(tmp_path, release_map)

    assert "NFP_Consensus_Median" in missing
    assert "NFP_Consensus_Good" in missing
    assert "NFP_Consensus_Mean" not in missing


def test_existing_unifier_cache_accepts_current_consensus_series(tmp_path):
    release_map = {
        pd.Timestamp("2026-04-01"): pd.Timestamp("2026-05-01"),
    }
    _write_unifier_snapshot(tmp_path, "2026-04-01", list(CONSENSUS_SERIES))

    missing = _missing_consensus_series_in_latest_snapshot(tmp_path, release_map)

    assert missing == set()
