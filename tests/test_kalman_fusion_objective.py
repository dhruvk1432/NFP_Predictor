"""Tests for the NSA LightGBM Optuna `kalman_fusion` objective mode.

Phase B of the fusion-aware tuning shift: instead of scoring each Optuna
trial by NSA-branch CV MAE, the inner CV runs the FULL fusion pipeline
(LGBM → adjustment → Kalman → composite vs SA revised). These tests
exercise the new path end-to-end on small synthetic data, without
requiring a real backtest run.

What we verify:
  1. The per-fold helper `_score_fold_kalman_fusion` runs without
     errors on a synthetic context and returns a finite score.
  2. A "good" model (predicting near the SA-revised target after
     adjustment) scores LOWER (better) than a "bad" model that
     predicts a constant far from the target.
  3. The `tune_hyperparameters` `objective_mode='kalman_fusion'` branch
     raises a clean error when `fusion_context` is missing.
  4. `_get_tuning_objective_mode('nsa', 'revised')` returns
     'kalman_fusion' when the config flag is on, and falls back to
     the legacy composite when the flag is off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Train import train_lightgbm_nfp as tln
from Train import hyperparameter_tuning as ht


class _ConstantModel:
    """Stub LightGBM-like model with .predict(X) returning a constant.

    The fusion-CV helper only calls `model.predict(X_tr_feats)` and
    `model.predict(X_val_feats)`. A simple stub lets us test the scoring
    logic without standing up real LightGBM training.
    """
    def __init__(self, value: float):
        self._value = float(value)

    def predict(self, X):
        n = len(X)
        return np.full(n, self._value, dtype=float)


def _build_synthetic_fusion_context(n_months: int = 60, seed: int = 0):
    """Construct a self-consistent fusion context.

    SA_revised[ds] = consensus[ds] + small_noise
    adj_pred[ds]   = 0  (zero adjustment for simplicity)
    NSA prediction in the "good" case is set to SA so champion=NSA+0≈SA.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_months, freq="MS")
    consensus_vals = rng.normal(150, 50, size=n_months)
    sa_vals = consensus_vals + rng.normal(0, 10, size=n_months)

    consensus_df = pd.DataFrame({
        "ds": dates,
        "consensus_pred": consensus_vals,
        "actual": sa_vals,
    })

    fusion_context = {
        "consensus_df": consensus_df,
        "consensus_by_ds": {pd.Timestamp(d): float(c)
                            for d, c in zip(dates, consensus_vals)},
        "sa_actuals_by_ds": {pd.Timestamp(d): float(s)
                             for d, s in zip(dates, sa_vals)},
        "adj_pred_by_ds": {pd.Timestamp(d): 0.0 for d in dates},
        "half_life_years": 3.0,
    }
    return fusion_context, dates, sa_vals


def test_fold_score_runs_and_is_finite():
    ctx, dates, sa_vals = _build_synthetic_fusion_context()
    n_tr = 40
    ds_tr = np.array(dates[:n_tr])
    ds_val = np.array(dates[n_tr:])

    # Stub features (the model ignores them — it returns a constant)
    X_tr_feats = pd.DataFrame({"x1": np.zeros(n_tr)})
    X_val_feats = pd.DataFrame({"x1": np.zeros(len(ds_val))})

    # "Good" predictor: predict the mean of validation SA actuals
    good_pred_value = float(np.mean(sa_vals[n_tr:]))
    score = ht._score_fold_kalman_fusion(
        model=_ConstantModel(good_pred_value),
        X_tr_feats=X_tr_feats, X_val_feats=X_val_feats,
        ds_tr=ds_tr, ds_val=ds_val,
        fusion_context=ctx,
    )
    assert np.isfinite(score), f"Expected finite score, got {score}"


def test_good_model_beats_bad_model():
    """A model whose NSA predictions land near SA_revised − adj_pred
    should score lower (better) than a model predicting a constant far
    from the target. We can't expect a strict inequality on every seed
    (Kalman fuses with consensus, which may dominate), so we average
    across seeds for a robust comparison."""
    n_better = 0
    n_seeds = 12
    for seed in range(n_seeds):
        ctx, dates, sa_vals = _build_synthetic_fusion_context(seed=seed)
        n_tr, n_val = 40, len(dates) - 40
        ds_tr = np.array(dates[:n_tr])
        ds_val = np.array(dates[n_tr:])
        X_tr_feats = pd.DataFrame({"x1": np.zeros(n_tr)})
        X_val_feats = pd.DataFrame({"x1": np.zeros(n_val)})

        # Good model: predict each validation row's SA actual directly.
        # We need .predict to return the per-row value — extend stub.
        class _RowModel:
            def __init__(self, tr_vals, val_vals):
                self.tr_vals = np.asarray(tr_vals, dtype=float)
                self.val_vals = np.asarray(val_vals, dtype=float)
                self._call_idx = 0

            def predict(self, X):
                # First call = training fold, second = validation fold.
                n = len(X)
                if self._call_idx == 0:
                    self._call_idx += 1
                    return self.tr_vals if len(self.tr_vals) == n else np.full(n, np.mean(self.tr_vals))
                return self.val_vals if len(self.val_vals) == n else np.full(n, np.mean(self.val_vals))

        good = _RowModel(sa_vals[:n_tr], sa_vals[n_tr:])
        bad = _ConstantModel(value=float(np.mean(sa_vals)) + 200.0)

        good_score = ht._score_fold_kalman_fusion(
            model=good, X_tr_feats=X_tr_feats, X_val_feats=X_val_feats,
            ds_tr=ds_tr, ds_val=ds_val, fusion_context=ctx,
        )
        bad_score = ht._score_fold_kalman_fusion(
            model=bad, X_tr_feats=X_tr_feats, X_val_feats=X_val_feats,
            ds_tr=ds_tr, ds_val=ds_val, fusion_context=ctx,
        )
        if good_score < bad_score:
            n_better += 1

    # The good model should be better in the strong majority of seeds.
    # We allow some slack because Kalman dampens NSA extremes.
    assert n_better >= int(0.75 * n_seeds), (
        f"Good model only beat bad model in {n_better}/{n_seeds} seeds"
    )


def test_tune_hyperparameters_raises_without_fusion_context():
    """If objective_mode='kalman_fusion' but fusion_context=None, the
    inner fold should raise — we want loud failure, not silent fallback,
    because the call site is supposed to gate this.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=80, freq="MS")
    X = pd.DataFrame({
        "ds": dates,
        "x1": rng.normal(size=80),
        "x2": rng.normal(size=80),
    })
    y = pd.Series(rng.normal(size=80))

    with pytest.raises(Exception):
        # The optuna study will run an objective that raises; depending
        # on how optuna propagates trial errors this may surface as a
        # ValueError or a different study-level exception. We only care
        # that calling without context does not silently produce a
        # finite result.
        ht.tune_hyperparameters(
            X, y, target_month=dates[-1],
            objective_mode="kalman_fusion",
            n_trials=1, timeout=15, use_huber_loss=False,
            fusion_context=None,
        )


def test_get_tuning_objective_mode_honors_flag():
    import Train.config as _cfg
    saved = _cfg.NSA_TUNE_USE_KALMAN_FUSION
    try:
        # When on: NSA → 'kalman_fusion'
        _cfg.NSA_TUNE_USE_KALMAN_FUSION = True
        # train_lightgbm_nfp imported the constant at module load time —
        # re-bind it for the test the same way.
        tln.NSA_TUNE_USE_KALMAN_FUSION = True
        assert tln._get_tuning_objective_mode("nsa", "revised") == "kalman_fusion"
        assert tln._get_tuning_objective_mode("sa", "revised") != "kalman_fusion"

        # When off: NSA falls back to legacy composite.
        tln.NSA_TUNE_USE_KALMAN_FUSION = False
        assert tln._get_tuning_objective_mode("nsa", "revised") != "kalman_fusion"
    finally:
        _cfg.NSA_TUNE_USE_KALMAN_FUSION = saved
        tln.NSA_TUNE_USE_KALMAN_FUSION = saved
