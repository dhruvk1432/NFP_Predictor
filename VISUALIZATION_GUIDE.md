# NFP Prediction Visualization Guide

## Enhanced Confidence Interval Plot

The prediction plot now includes **multi-level confidence interval shading** to help you visualize prediction uncertainty and evaluate model performance.

### Plot Location

```
_output/backtest_results/nsa/predictions_vs_actual.png
```

### Visual Elements

#### 1. **Confidence Interval Shading** (Gradient Effect)
The plot displays three nested confidence intervals with progressively darker shading:

- **95% CI** (Lightest blue, `alpha=0.15`): Outermost band
  - Expected to contain ~95% of actual values
  - Current coverage: **85.7%** (30/35 predictions)

- **80% CI** (Medium blue, `alpha=0.25`): Middle band
  - Expected to contain ~80% of actual values
  - Current coverage: **77.1%** (27/35 predictions)

- **50% CI** (Darkest blue, `alpha=0.35`): Innermost band
  - Expected to contain ~50% of actual values
  - Current coverage: **48.6%** (17/35 predictions)

The gradient effect creates a visual representation of confidence: **darker = more confident**.

#### 2. **Data Lines**

- **Black line with circles** (`●`): Actual NFP values
  - Only shown for historical data (Jan 2023 - Nov 2025)
  - Not shown for future predictions (Dec 2025)

- **Blue line with squares** (`■`): Model predictions
  - Shown for all months including historical and future

- **Red dashed line with diamonds** (`◆`): Future predictions
  - Highlights December 2025 prediction (-100k)
  - Uses dashed line to indicate uncertainty

#### 3. **Reference Lines**

- **Gray dashed horizontal line**: Zero line (y=0)
  - Helps identify whether NFP is positive (job gains) or negative (job losses)

### How to Interpret the Plot

#### Good Model Performance Indicators:
1. **Actual values fall within shaded bands** - Shows calibrated uncertainty
2. **Prediction line tracks actual line** - Shows accurate point estimates
3. **Narrower bands** - Indicates higher model confidence
4. **Consistent coverage** - Coverage metrics match expected percentiles

#### Red Flags to Watch For:
1. **Actual values consistently outside bands** - Underfitted or miscalibrated intervals
2. **Asymmetric errors** - Model systematically over/under-predicting
3. **Wider bands in certain periods** - Model uncertain during specific conditions
4. **Coverage metrics far from expected** - Interval calibration issues

### Current Model Performance

Based on 35 historical predictions (Jan 2023 - Nov 2025):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 230.48 | Average prediction error magnitude |
| **MAE** | 163.55 | Typical absolute error |
| **Coverage 50%** | 48.6% | ✓ Well calibrated (close to 50%) |
| **Coverage 80%** | 77.1% | ✓ Well calibrated (close to 80%) |
| **Coverage 95%** | 85.7% | ✓ Reasonable (slightly under 95%) |

### December 2025 Prediction

The plot shows the future prediction for December 2025:

- **Point Prediction**: -100,000 jobs (loss)
- **80% Confidence Interval**: [-453,000, +339,000]
- **Interpretation**: Model predicts job losses, but with high uncertainty (wide interval crosses zero)

### Regenerating the Plot

To regenerate the plot after model updates:

```bash
python regenerate_plot.py
```

This script:
1. Reads the latest predictions from `_output/backtest_results/nsa/backtest_results_nsa.csv`
2. Applies enhanced visualization with CI shading
3. Saves updated plot to the same location

### Technical Details

**Implementation**: [`Train/backtest_results.py`](Train/backtest_results.py)

The `plot_predictions_vs_actual()` function:
- Uses matplotlib's `fill_between()` for shaded regions
- Applies z-order layering for proper visual stacking
- Separates historical and future predictions automatically
- Supports any dataset with columns: `date`, `pred`, `actual`, `lower_50`, `upper_50`, `lower_80`, `upper_80`, `lower_95`, `upper_95`

**Confidence Interval Calculation**: [`Train/train_lightgbm_nfp.py:1508-1545`](Train/train_lightgbm_nfp.py#L1508-L1545)

Intervals are computed using residual-based quantiles:
1. Calculate residuals from recent training predictions (last 12 months)
2. Compute percentiles: 2.5%, 10%, 25%, 75%, 90%, 97.5%
3. Add/subtract from point prediction to create symmetric intervals

### Next Steps for Improvement

If coverage metrics are consistently off:

1. **Recalibrate intervals** - Adjust percentile calculations
2. **Use conformal prediction** - More rigorous uncertainty quantification
3. **Add asymmetric intervals** - Account for skewed error distribution
4. **Conditional intervals** - Width varies by prediction characteristics

---

**Last Updated**: 2026-01-08
**Model Version**: LightGBM NSA Walk-Forward Backtest (36 months)
