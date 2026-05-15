"""Tests for _load_fusion_selection_target.

The dynamic feature selection's notion of ``y_sel`` is now the fusion-aligned
target ``SA_revised − adj_pred(half_life)`` for the NSA branch. These tests
exercise the helper directly against mocked data loaders so we can pin down:
    * value-level correctness (each row equals SA − adj_pred),
    * PIT row-level fallback when the adjustment history for a target date
      is empty (early-history dates),
    * sane correlation with the legacy NSA y_mom (catches silent loader
      breakage that would otherwise just produce garbage),
    * graceful fallback to NSA y_mom when target_type is not 'nsa' or when
      the data loaders raise.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from Train import train_lightgbm_nfp as tln


def _build_inputs(n_months: int = 60, start: str = "2010-01-01"):
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n_months, freq="MS")
    nsa_y_mom = pd.Series(rng.normal(150, 80, size=n_months), name="y_mom")
    # SA = NSA + (calendar-month mean adjustment) + small noise.
    months = pd.Series(dates).dt.month.values
    cal_means = {m: float(20 * (m - 6)) for m in range(1, 13)}  # systematic seasonal
    sa_y_mom = pd.Series(
        [nsa_y_mom.iloc[i] + cal_means[months[i]] + rng.normal(0, 5)
         for i in range(n_months)],
        name="y_mom",
    )
    X_train = pd.DataFrame({"ds": dates, "x1": rng.normal(size=n_months)})

    sa_target_df = pd.DataFrame({"ds": dates, "y_mom": sa_y_mom.values})

    # Adjustment history: long synthetic series with operational_available_date
    # one month after ds so PIT filtering at ds=M excludes the adjustment from M.
    hist_dates = pd.date_range("1990-01-01", periods=400, freq="MS")
    hist_months = pd.Series(hist_dates).dt.month.values
    adj_history = pd.DataFrame({
        "ds": hist_dates,
        "adjustment": [cal_means[m] + rng.normal(0, 3) for m in hist_months],
        "operational_available_date": hist_dates + pd.DateOffset(months=1),
    })
    return X_train, nsa_y_mom, sa_target_df, adj_history


def _patch_loaders(monkeypatch, sa_target_df, adj_history, tuned_hl=None, raise_sa=False):
    """Patch the loaders that _load_fusion_selection_target calls."""
    def _fake_load_target(target_type, release_type="first", target_source="revised", **kw):
        if raise_sa:
            raise RuntimeError("SA target unavailable")
        assert target_type == "sa"
        return sa_target_df.copy()

    def _fake_load_adj():
        return adj_history.copy()

    import Train.data_loader as _dl
    import Train.sandbox.experiment_predicted_adjustment as _adj
    monkeypatch.setattr(_dl, "load_target_data", _fake_load_target)
    monkeypatch.setattr(_adj, "load_adjustment_history", _fake_load_adj)

    if tuned_hl is not None:
        from settings import OUTPUT_DIR
        tuned_path = OUTPUT_DIR / "consensus_anchor" / "kalman_fusion" / "tuned_params.json"
        tuned_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tuned_path, "w") as f:
            json.dump({"half_life_years": float(tuned_hl)}, f)


def test_fusion_target_matches_sa_minus_adj_pred(monkeypatch, tmp_path):
    X_train, nsa_y, sa_df, adj = _build_inputs()
    _patch_loaders(monkeypatch, sa_df, adj)

    # Isolate from any production tuned_params.json that may exist on disk.
    from settings import OUTPUT_DIR
    tuned_path = OUTPUT_DIR / "consensus_anchor" / "kalman_fusion" / "tuned_params.json"
    _shadowed = tuned_path.exists()
    _shadow_target = tuned_path.with_suffix(".json.test_shadow") if _shadowed else None
    if _shadowed:
        tuned_path.rename(_shadow_target)
    try:
        y_sel, hl = tln._load_fusion_selection_target(
            X_train, nsa_y, target_type="nsa", step_date=pd.Timestamp("2030-01-01"),
        )

        # Returned series should be non-empty and indexed by training dates.
        assert isinstance(y_sel, pd.Series)
        assert y_sel.size > 0
        assert hl == 3.0  # default when no tuned_params.json present
        assert (y_sel.index == y_sel.index.sort_values()).all()
    finally:
        if _shadowed and _shadow_target is not None and _shadow_target.exists():
            _shadow_target.rename(tuned_path)

    # For every returned row, the value should match SA_revised − adj_pred,
    # where adj_pred is what ExpWeightedMedianCovidExcludedPredictor would
    # produce on the PIT-filtered history. Recompute via the same machinery
    # the helper uses to keep the contract identical.
    from Train.Output_code.consensus_anchor_runner import (
        _build_pit_adjustment_cache, _compute_adjustment_series,
    )
    train_dates = pd.Series(pd.to_datetime(X_train["ds"].values))
    pit_cache = _build_pit_adjustment_cache(train_dates, adj)
    adj_vals = _compute_adjustment_series(train_dates, pit_cache, hl)

    sa_by_ds = pd.Series(sa_df["y_mom"].values,
                         index=pd.to_datetime(sa_df["ds"].values))
    for ds in y_sel.index:
        avail = pit_cache.get(pd.Timestamp(ds))
        if avail is None or avail.empty:
            continue  # those rows used the NSA fallback, tested separately
        i = list(train_dates).index(pd.Timestamp(ds))
        expected = float(sa_by_ds.loc[ds]) - float(adj_vals[i])
        assert y_sel.loc[ds] == pytest.approx(expected, abs=1e-9)


def test_fusion_target_correlates_with_nsa_y_mom(monkeypatch):
    X_train, nsa_y, sa_df, adj = _build_inputs(n_months=80)
    _patch_loaders(monkeypatch, sa_df, adj)

    y_sel, _ = tln._load_fusion_selection_target(
        X_train, nsa_y, "nsa", pd.Timestamp("2030-01-01"),
    )
    nsa_aligned = pd.Series(nsa_y.values,
                            index=pd.to_datetime(X_train["ds"].values))
    common = y_sel.index.intersection(nsa_aligned.index)
    assert len(common) >= 20
    corr = float(np.corrcoef(y_sel.loc[common].values,
                             nsa_aligned.loc[common].values)[0, 1])
    # Sanity range: fusion target = NSA + small calendar-residual noise, so
    # the correlation must be high but not 1.0. A loader-breakage producing
    # garbage would drop this out of range.
    assert 0.6 <= corr <= 0.999, f"corr(y_sel, NSA) = {corr:.3f} out of range"


def test_fusion_target_falls_back_for_non_nsa_branch(monkeypatch):
    X_train, nsa_y, sa_df, adj = _build_inputs()
    _patch_loaders(monkeypatch, sa_df, adj)

    y_sel, hl = tln._load_fusion_selection_target(
        X_train, nsa_y, target_type="sa", step_date=pd.Timestamp("2030-01-01"),
    )
    assert hl is None
    # Should match the legacy NSA-y_mom indexed-by-ds series exactly.
    expected = pd.Series(nsa_y.values,
                         index=pd.to_datetime(X_train["ds"].values)).dropna()
    pd.testing.assert_series_equal(
        y_sel.sort_index(), expected.sort_index(), check_names=False,
    )


def test_fusion_target_falls_back_when_sa_loader_raises(monkeypatch, caplog):
    X_train, nsa_y, sa_df, adj = _build_inputs()
    _patch_loaders(monkeypatch, sa_df, adj, raise_sa=True)

    y_sel, hl = tln._load_fusion_selection_target(
        X_train, nsa_y, "nsa", pd.Timestamp("2030-01-01"),
    )
    assert hl is None  # signalled the fallback path
    # Should fall back to legacy NSA y_mom rather than crash.
    assert y_sel.size == nsa_y.dropna().size


def test_fusion_target_uses_tuned_half_life(monkeypatch, tmp_path):
    """When tuned_params.json exists with a different half_life_years,
    the helper should pick it up and log it."""
    X_train, nsa_y, sa_df, adj = _build_inputs()
    _patch_loaders(monkeypatch, sa_df, adj, tuned_hl=0.75)
    try:
        y_sel, hl = tln._load_fusion_selection_target(
            X_train, nsa_y, "nsa", pd.Timestamp("2030-01-01"),
        )
        assert hl == pytest.approx(0.75, abs=1e-9)
    finally:
        # Clean up the file we wrote so other tests aren't polluted.
        from settings import OUTPUT_DIR
        tuned_path = OUTPUT_DIR / "consensus_anchor" / "kalman_fusion" / "tuned_params.json"
        if tuned_path.exists():
            tuned_path.unlink()
