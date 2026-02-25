"""Tests for keep-rule enforcement logic in run_expanding_window_backtest()."""

import numpy as np
import pandas as pd
import pytest


def _make_results_df(n_rows=20, model_mae_scale=100, baseline_mae_scale=80):
    """Create a synthetic results_df mimicking backtest output.

    Args:
        n_rows: Number of OOS months.
        model_mae_scale: Scale of model errors (higher = worse).
        baseline_mae_scale: Scale of baseline errors (higher = worse).
    """
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="MS")
    actual = rng.randn(n_rows) * 200 + 150

    model_pred = actual + rng.randn(n_rows) * model_mae_scale
    baseline_pred = actual + rng.randn(n_rows) * baseline_mae_scale

    df = pd.DataFrame({
        'ds': dates,
        'actual': actual,
        'predicted': model_pred,
        'error': actual - model_pred,
        'baseline_last_y': baseline_pred,
        'baseline_last_y_error': actual - baseline_pred,
    })
    return df


def _run_keep_rule(results_df, window_m=12, tolerance=0.0, action='skip_save'):
    """Extract and run the keep-rule logic (mirrors train_lightgbm_nfp.py)."""
    backtest_only = results_df[~results_df['error'].isna()].copy()

    if len(backtest_only) < window_m:
        return  # keep-rule skipped

    trailing = backtest_only.iloc[-window_m:]
    trailing_indices = trailing.index

    trailing_model_mae = np.mean(np.abs(trailing['error']))

    baseline_error_cols = [c for c in trailing.columns
                           if c.startswith('baseline_') and c.endswith('_error')]

    for col in baseline_error_cols:
        bname = col.replace('_error', '')
        valid_errors = trailing[col].dropna()
        if valid_errors.empty:
            continue

        trailing_baseline_mae = np.mean(np.abs(valid_errors))
        degradation = trailing_model_mae - trailing_baseline_mae

        if degradation > tolerance:
            if action == 'fail':
                raise RuntimeError(
                    f"Keep-rule failed: model underperforms '{bname}' by "
                    f"{degradation:.2f} MAE."
                )
            elif action == 'fallback_to_baseline':
                results_df.loc[trailing_indices, 'predicted'] = results_df.loc[trailing_indices, bname]
                results_df.loc[trailing_indices, 'error'] = (
                    results_df.loc[trailing_indices, 'actual']
                    - results_df.loc[trailing_indices, 'predicted']
                )
                results_df.loc[trailing_indices, 'keep_rule_fallback'] = True
            elif action == 'skip_save':
                results_df.attrs['skip_save'] = True
            break


def test_skip_save_flag_set():
    """When model MAE > baseline MAE, skip_save should be True."""
    df = _make_results_df(n_rows=20, model_mae_scale=150, baseline_mae_scale=50)
    _run_keep_rule(df, window_m=12, tolerance=0.0, action='skip_save')
    assert df.attrs.get('skip_save', False) is True


def test_keep_rule_pass_no_flag():
    """When model beats baseline, skip_save should not be set."""
    df = _make_results_df(n_rows=20, model_mae_scale=30, baseline_mae_scale=150)
    _run_keep_rule(df, window_m=12, tolerance=0.0, action='skip_save')
    assert df.attrs.get('skip_save', False) is False


def test_fail_action_raises():
    """KEEP_RULE_ACTION='fail' should raise RuntimeError."""
    df = _make_results_df(n_rows=20, model_mae_scale=150, baseline_mae_scale=50)
    with pytest.raises(RuntimeError, match="Keep-rule failed"):
        _run_keep_rule(df, window_m=12, tolerance=0.0, action='fail')


def test_fallback_replaces_predictions_by_index():
    """Fallback should replace predictions with baseline values in trailing window."""
    df = _make_results_df(n_rows=20, model_mae_scale=150, baseline_mae_scale=50)
    baseline_values_before = df['baseline_last_y'].iloc[-12:].copy()

    _run_keep_rule(df, window_m=12, tolerance=0.0, action='fallback_to_baseline')

    # Predictions should now equal baseline values for trailing rows
    trailing_predicted = df['predicted'].iloc[-12:]
    pd.testing.assert_series_equal(
        trailing_predicted.reset_index(drop=True),
        baseline_values_before.reset_index(drop=True),
        check_names=False,
    )

    # keep_rule_fallback column should be True for those rows
    assert df['keep_rule_fallback'].iloc[-12:].all()

    # Error should be recomputed: actual - predicted (= actual - baseline)
    recomputed = df['actual'].iloc[-12:] - df['predicted'].iloc[-12:]
    pd.testing.assert_series_equal(
        df['error'].iloc[-12:].reset_index(drop=True),
        recomputed.reset_index(drop=True),
        check_names=False,
    )


def test_keep_rule_skipped_insufficient_months():
    """If fewer OOS months than KEEP_RULE_WINDOW_M, no flag should be set."""
    df = _make_results_df(n_rows=5, model_mae_scale=200, baseline_mae_scale=50)
    _run_keep_rule(df, window_m=12, tolerance=0.0, action='skip_save')
    assert df.attrs.get('skip_save', False) is False


def test_tolerance_allows_slight_degradation():
    """Model can be slightly worse than baseline if within tolerance."""
    rng = np.random.RandomState(42)
    n = 20
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    actual = np.ones(n) * 100

    # Model errors: always +5 -> MAE = 5
    # Baseline errors: always +3 -> MAE = 3
    # Degradation = 5 - 3 = 2. With tolerance=3, should NOT trigger.
    df = pd.DataFrame({
        'ds': dates,
        'actual': actual,
        'predicted': actual - 5,
        'error': np.full(n, 5.0),
        'baseline_last_y': actual - 3,
        'baseline_last_y_error': np.full(n, 3.0),
    })
    _run_keep_rule(df, window_m=12, tolerance=3.0, action='skip_save')
    assert df.attrs.get('skip_save', False) is False
