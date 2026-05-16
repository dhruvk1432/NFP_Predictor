import pandas as pd

from utils.exact_feature_pruning import (
    drop_non_interesting_long,
    drop_non_interesting_wide,
    exact_feature_prune_schema_version,
    filter_feature_names,
)


def _write_manifest(path, rows):
    pd.DataFrame(rows, columns=["source", "feature"]).to_csv(path, index=False)


def test_filter_feature_names_drops_only_exact_manifest_matches(tmp_path):
    manifest = tmp_path / "non_interesting.csv"
    _write_manifest(
        manifest,
        [
            {"source": "ADP", "feature": "ADP_actual_pct_chg_lag_3m"},
            {"source": "Calendar", "feature": "cal_month_sin"},
        ],
    )

    kept = filter_feature_names(
        "ADP",
        ["ADP_actual", "ADP_actual_pct_chg_lag_3m", "ADP_actual_pct_chg_lag_6m"],
        manifest_path=manifest,
    )

    assert kept == ["ADP_actual", "ADP_actual_pct_chg_lag_6m"]


def test_filter_feature_names_prefers_allow_manifest_when_present(tmp_path):
    deny_manifest = tmp_path / "non_interesting.csv"
    allow_manifest = tmp_path / "usable.csv"
    _write_manifest(deny_manifest, [{"source": "ADP", "feature": "ADP_actual_pct_chg_lag_6m"}])
    _write_manifest(
        allow_manifest,
        [
            {"source": "ADP", "feature": "ADP_actual"},
            {"source": "ADP", "feature": "ADP_actual_pct_chg_lag_3m"},
        ],
    )

    kept = filter_feature_names(
        "ADP",
        ["ADP_actual", "ADP_actual_pct_chg_lag_3m", "ADP_actual_pct_chg_lag_6m"],
        manifest_path=deny_manifest,
        allow_manifest_path=allow_manifest,
    )

    assert kept == ["ADP_actual", "ADP_actual_pct_chg_lag_3m"]


def test_drop_non_interesting_wide_preserves_metadata(tmp_path):
    manifest = tmp_path / "non_interesting.csv"
    _write_manifest(manifest, [{"source": "Calendar", "feature": "cal_month_sin"}])
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "snapshot_date": pd.to_datetime(["2024-02-02"]),
            "cal_month_sin": [0.5],
            "cal_month_cos": [0.8],
        }
    )

    pruned = drop_non_interesting_wide(df, "Calendar", manifest_path=manifest)

    assert list(pruned.columns) == ["date", "snapshot_date", "cal_month_cos"]


def test_drop_non_interesting_long_drops_only_matching_source(tmp_path):
    manifest = tmp_path / "non_interesting.csv"
    _write_manifest(manifest, [{"source": "NOAA", "feature": "NOAA_Human_Impact_Index_pct_chg"}])
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "series_name": ["NOAA_Human_Impact_Index_pct_chg", "NOAA_Economic_Damage_Index"],
            "value": [1.0, 2.0],
        }
    )

    assert drop_non_interesting_long(df, "NOAA", manifest_path=manifest)["series_name"].tolist() == [
        "NOAA_Economic_Damage_Index"
    ]
    assert drop_non_interesting_long(df, "Prosper", manifest_path=manifest)["series_name"].tolist() == df[
        "series_name"
    ].tolist()


def test_schema_version_tracks_manifest_contents(tmp_path):
    deny_manifest = tmp_path / "deny.csv"
    allow_manifest = tmp_path / "allow.csv"
    _write_manifest(deny_manifest, [{"source": "ADP", "feature": "A"}])
    _write_manifest(allow_manifest, [{"source": "ADP", "feature": "B"}])
    before = exact_feature_prune_schema_version(deny_manifest, allow_manifest)

    _write_manifest(deny_manifest, [{"source": "ADP", "feature": "C"}])
    after = exact_feature_prune_schema_version(deny_manifest, allow_manifest)

    assert before != after
