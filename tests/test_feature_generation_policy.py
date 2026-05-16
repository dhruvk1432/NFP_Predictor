import pandas as pd

from utils.feature_generation_policy import (
    RETENTION_USE_CASES,
    filter_feature_names,
    filter_long_features,
    reset_feature_policy_audit,
    should_generate_intermediate_feature,
)
from utils.transforms import add_pct_change_copies, compute_all_features


def _write_manifest(path, rows):
    pd.DataFrame(rows, columns=["source", "feature"]).to_csv(path, index=False)


def test_exact_deny_wins_over_protected_signal(monkeypatch, tmp_path):
    allow = tmp_path / "allow.csv"
    deny = tmp_path / "deny.csv"
    _write_manifest(allow, [{"source": "DerivedControls", "feature": "nfp_nsa_accel_lag6"}])
    _write_manifest(deny, [{"source": "DerivedControls", "feature": "nfp_nsa_accel_lag6"}])
    monkeypatch.setenv("NFP_FEATURE_POLICY", "usable_exact")
    monkeypatch.setenv("NFP_FEATURE_POLICY_ALLOW", str(allow))
    monkeypatch.setenv("NFP_FEATURE_POLICY_DENY", str(deny))
    reset_feature_policy_audit()

    kept = filter_feature_names(
        "DerivedControls",
        ["nfp_nsa_accel_lag6", "nfp_nsa_accel_vol_3m"],
    )

    assert kept == ["nfp_nsa_accel_vol_3m"]


def test_exact_allow_preserves_noaa_feature_while_blocking_source_family(monkeypatch, tmp_path):
    allow = tmp_path / "allow.csv"
    deny = tmp_path / "deny.csv"
    _write_manifest(allow, [{"source": "NOAA", "feature": "NOAA_Human_Impact_Index_chg_3m_lag_1m"}])
    _write_manifest(deny, [])
    monkeypatch.setenv("NFP_FEATURE_POLICY", "usable_exact")
    monkeypatch.setenv("NFP_FEATURE_POLICY_ALLOW", str(allow))
    monkeypatch.setenv("NFP_FEATURE_POLICY_DENY", str(deny))
    reset_feature_policy_audit()

    dates = pd.date_range("2020-01-01", periods=8, freq="MS")
    df = pd.DataFrame(
        {
            "date": dates,
            "series_name": "NOAA_Human_Impact_Index",
            "value": range(8),
        }
    )
    result = compute_all_features(df, lean=True, source_name="NOAA")

    assert sorted(result["series_name"].unique()) == ["NOAA_Human_Impact_Index_chg_3m_lag_1m"]


def test_audit_mode_keeps_blocked_features(monkeypatch, tmp_path):
    allow = tmp_path / "allow.csv"
    deny = tmp_path / "deny.csv"
    _write_manifest(allow, [])
    _write_manifest(deny, [{"source": "ADP", "feature": "ADP_actual_pct_chg_lag_3m"}])
    monkeypatch.setenv("NFP_FEATURE_POLICY", "audit")
    monkeypatch.setenv("NFP_FEATURE_POLICY_ALLOW", str(allow))
    monkeypatch.setenv("NFP_FEATURE_POLICY_DENY", str(deny))
    reset_feature_policy_audit()

    kept = filter_feature_names("ADP", ["ADP_actual", "ADP_actual_pct_chg_lag_3m"])

    assert kept == ["ADP_actual", "ADP_actual_pct_chg_lag_3m"]


def test_strict_blocks_unknown_source_unless_allowlisted(monkeypatch, tmp_path):
    allow = tmp_path / "allow.csv"
    deny = tmp_path / "deny.csv"
    _write_manifest(allow, [{"source": "Unknown", "feature": "mystery_signal"}])
    _write_manifest(deny, [])
    monkeypatch.setenv("NFP_FEATURE_POLICY", "strict")
    monkeypatch.setenv("NFP_FEATURE_POLICY_ALLOW", str(allow))
    monkeypatch.setenv("NFP_FEATURE_POLICY_DENY", str(deny))
    reset_feature_policy_audit()

    kept = filter_feature_names("Unknown", ["mystery_signal", "other_mystery_signal"])

    assert kept == ["mystery_signal"]


def test_unstable_pct_change_intermediate_is_not_materialized(monkeypatch, tmp_path):
    allow = tmp_path / "allow.csv"
    deny = tmp_path / "deny.csv"
    _write_manifest(allow, [])
    _write_manifest(deny, [])
    monkeypatch.setenv("NFP_FEATURE_POLICY", "usable_exact")
    monkeypatch.setenv("NFP_FEATURE_POLICY_ALLOW", str(allow))
    monkeypatch.setenv("NFP_FEATURE_POLICY_DENY", str(deny))
    reset_feature_policy_audit()

    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
            "series_name": "Treasury_10Y_close",
            "value": [1.0, 1.2, 1.1, 1.3],
        }
    )
    out = add_pct_change_copies(df, source_name="Futures")

    assert "Treasury_10Y_close_pct_chg" not in set(out["series_name"])
    assert not should_generate_intermediate_feature("Futures", "Treasury_10Y_close_pct_chg")


def test_filter_long_features_drops_family_blocked_source(monkeypatch, tmp_path):
    allow = tmp_path / "allow.csv"
    deny = tmp_path / "deny.csv"
    _write_manifest(allow, [])
    _write_manifest(deny, [])
    monkeypatch.setenv("NFP_FEATURE_POLICY", "usable_exact")
    monkeypatch.setenv("NFP_FEATURE_POLICY_ALLOW", str(allow))
    monkeypatch.setenv("NFP_FEATURE_POLICY_DENY", str(deny))
    reset_feature_policy_audit()

    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="MS"),
            "series_name": ["Prosper_A", "Prosper_B"],
            "value": [1.0, 2.0],
        }
    )

    assert filter_long_features(df, "Prosper").empty


def test_policy_retention_scope_covers_side_model_use_cases():
    assert {
        "acceleration",
        "hmm",
        "nsa_prediction",
        "kalman",
        "sa_prediction",
        "other_time_series",
    } <= RETENTION_USE_CASES
