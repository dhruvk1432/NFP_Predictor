# Master Snapshot Analysis Report
## Comprehensive Feature Audit for NFP Prediction Pipeline

**Date**: 2026-02-09
**Snapshot analyzed**: 2025-12 (latest)
**Total unique series**: 1,106 (553 original + 553 symlog copies)

---

## Executive Summary

| Verdict | Count | % | Description |
|---------|-------|---|-------------|
| **KEEP** | 292 | 52.8% | Essential or statistically strong |
| **CONDITIONAL** | 247 | 44.7% | Keep only if model performance improves |
| **DROP** | 14 | 2.5% | Remove from pipeline |

### Verdict by Source

| Source | KEEP | CONDITIONAL | DROP | Total |
|--------|------|-------------|------|-------|
| FRED Exog | 160 | 74 | 0 | 234 |
| Unifier | 92 | 29 | 10 | 131 |
| Prosper | 38 | 143 | 4 | 185 |
| ADP | 1 | 0 | 0 | 1 |
| NOAA | 1 | 1 | 0 | 2 |

### Transform Recommendations

| Recommendation | Count | Meaning |
|----------------|-------|---------|
| RAW_ONLY | 442 | Raw data is fine, symlog adds nothing |
| SYMLOG_PREFERRED | 61 | Symlog significantly improves distribution |
| BOTH | 50 | Keep both raw and symlog versions |

---

## Top 30 Most Correlated to NSA NFP MoM (Primary Target)

| # | Series | Source | r | Importance | Verdict |
|---|--------|--------|---|------------|---------|
| 1 | CCNSA_monthly_avg_diff_zscore_12m | FRED | **-0.643** | CRITICAL | KEEP |
| 2 | AHE_Private_diff | Unifier | **-0.634** | MEDIUM | KEEP |
| 3 | CCNSA_monthly_avg_pct_chg_zscore_12m | FRED | **-0.632** | CRITICAL | KEEP |
| 4 | ADP_actual | ADP | **0.608** | CRITICAL | KEEP |
| 5 | Weekly_Econ_Index_monthly_max_symlog_diff | FRED | **0.602** | HIGH | KEEP |
| 6 | Industrial_Production_symlog_diff | Unifier | **0.559** | MEDIUM | KEEP |
| 7 | "I know people laid off" Females_pct_chg | Prosper | **-0.553** | LOW_MED | KEEP |
| 8 | Industrial_Production_diff | Unifier | **0.550** | MEDIUM | KEEP |
| 9 | Weekly_Econ_Index_monthly_max_diff | FRED | **0.541** | HIGH | KEEP |
| 10 | Challenger_Job_Cuts_diff | Unifier | **-0.531** | HIGH | KEEP |
| 11 | Oil_Prices_volatility | FRED | **-0.461** | MEDIUM | KEEP |
| 12 | Challenger_Job_Cuts | Unifier | **-0.437** | HIGH | KEEP |
| 13 | Oil_Prices_volatility_diff | FRED | **-0.410** | MEDIUM | KEEP |
| 14 | Prosper Spending Forecast Males_diff_zscore_12m | Prosper | **0.388** | LOW | KEEP |
| 15 | CCNSA_max_spike_diff_zscore_3m | FRED | **-0.357** | CRITICAL | KEEP |

## Top 20 Most Correlated to SA NFP MoM (Final Target)

| # | Series | Source | r | NSA r |
|---|--------|--------|---|-------|
| 1 | AHE_Private_diff | Unifier | **-0.780** | 0.634 |
| 2 | ADP_actual | ADP | **0.762** | 0.608 |
| 3 | Industrial_Production_symlog_diff | Unifier | **0.753** | 0.559 |
| 4 | "I know people laid off" Females_pct_chg | Prosper | **-0.704** | 0.553 |
| 5 | Weekly_Econ_Index_monthly_max_symlog_diff | FRED | **0.682** | 0.602 |
| 6 | Oil_Prices_volatility | FRED | **-0.677** | 0.461 |
| 7 | Challenger_Job_Cuts_diff | Unifier | **-0.587** | 0.531 |
| 8 | Oil_Prices_volatility_diff | FRED | **-0.569** | 0.410 |
| 9 | "More layoffs expected" Females_pct_chg | Prosper | **-0.519** | 0.301 |
| 10 | ISM_Manufacturing_Index_symlog_pct_chg | Unifier | **0.396** | 0.267 |

---

## Source-by-Source Deep Dive

### 1. FRED Exogenous (234 series: 160 KEEP, 74 CONDITIONAL, 0 DROP)

**The backbone of the model.** Contains financial markets, credit, claims, and real-time economic indicators. Zero series recommended for removal.

#### CCSA/CCNSA (Initial Claims) - CRITICAL
- **What it is**: Weekly initial unemployment claims, NSA and SA versions
- **Why it matters**: Direct inverse indicator of hiring. People filing = people losing jobs. The #1 correlated predictor for NSA target (r = -0.643).
- **Best transforms**: `_diff_zscore_12m` (r = 0.643), `_pct_chg_zscore_12m` (r = 0.632)
- **Recommendation**: **KEEP ALL transforms** - every CCNSA/CCSA variant is useful
- **Raw vs SymLog**: Raw is better for z-score transforms (already normalized). SymLog helps the `_monthly_avg` level (kurtosis 49 -> 7)
- **Specific notes**: `CCSA_monthly_avg` has skew=6.3, kurt=49.4 (COVID spike) - use symlog for this one; the z-score versions are already well-behaved

#### VIX (Volatility Index) - HIGH
- **What it is**: CBOE Volatility Index ("fear gauge")
- **Why it matters**: Market fear precedes hiring freezes. VIX_panic_regime (binary) correlates -0.40 with SA target
- **Best transforms**: Binary regimes (panic, high) are the most useful
- **Max correlation**: 0.287 (NSA), 0.404 (SA)
- **Recommendation**: **KEEP binaries + key z-scores, DROP most individual transforms**
- **Raw vs SymLog**: VIX_max_5d_spike is heavily skewed (use symlog). Z-scores are fine raw.
- Keep: `VIX_panic_regime`, `VIX_high_regime`, `VIX_max_5d_spike` (with symlog), `VIX_volatility_diff_zscore_12m`
- Drop consideration: `VIX_mean`, `VIX_max`, `VIX_30d_spike` transforms add little incremental value (r < 0.03)

#### SP500 - HIGH
- **What it is**: S&P 500 stock market index derived features
- **Why it matters**: Wealth effect, business confidence, forward-looking
- **Best transforms**: `SP500_bear_market` (binary, r_sa=0.20), `SP500_crash_month` (binary), `SP500_max_drawdown` (level)
- **Max correlation**: 0.287 (NSA via VIX_panic_regime co-movement), individual SP500 features weaker
- **Recommendation**: **KEEP binary flags + max_drawdown + monthly_return, DROP granular transforms**
- **Raw vs SymLog**: Monthly return is fine raw. Best_day/worst_day are heavily skewed - symlog helps but correlations are very weak anyway
- Most SP500 z-score transforms have near-zero correlation (r < 0.06) - noise

#### Credit Spreads - HIGH
- **What it is**: Corporate bond spreads (investment grade vs high yield)
- **Why it matters**: Tightening credit = less hiring. Leading indicator.
- **Best transforms**: `Credit_Spreads_monthly_chg_diff` (r=0.227 NSA), `Credit_Spreads_acceleration` level (r=0.134/0.269 SA)
- **Recommendation**: **KEEP level + monthly_chg_diff + acceleration. CONDITIONAL on z-scores**
- **Raw vs SymLog**: Acceleration has moderate kurtosis (8.4) - symlog improves it slightly. Raw z-scores are fine.

#### Yield Curve (10Y-2Y) - HIGH
- **What it is**: 10-year minus 2-year Treasury yield spread
- **Why it matters**: Classic recession predictor with 12-month lead. Inversion = recession signal.
- **Best transforms**: `Yield_Curve_avg` (level, r=0.093/0.051), weak overall
- **Max correlation**: Only 0.093 NSA, 0.073 SA - surprisingly weak in contemporaneous correlation
- **Recommendation**: **KEEP level + monthly_chg + vol_of_changes, but understand its value is as a LEADING indicator** (12-month lag, not captured by level correlation). The pipeline already uses lagged versions.
- **Raw vs SymLog**: All well-behaved, raw is fine

#### Oil Prices - MEDIUM
- **What it is**: WTI crude oil price and derived volatility/crash indicators
- **Why it matters**: Input cost for transportation/manufacturing. Volatility is the key signal.
- **Best transforms**: `Oil_Prices_volatility` (r=0.461 NSA, 0.677 SA!) - extremely strong for SA target
- **Recommendation**: **KEEP volatility + volatility_diff. CONDITIONAL on mean, crash, zscore_min**
- **Raw vs SymLog**: Oil_Prices_volatility is EXTREMELY leptokurtic (kurt=107, skew=7.9) - **SYMLOG ESSENTIAL** (reduces to kurt=3.9)
- The `_volatility_diff` also needs symlog (kurt=136)

#### Weekly Economic Index - HIGH
- **What it is**: NY Fed real-time GDP tracking index
- **Why it matters**: Most timely broad economic indicator
- **Best transforms**: `monthly_max_symlog_diff` (r=0.602 NSA, 0.682 SA) - one of the best predictors overall
- **Recommendation**: **KEEP monthly_max (symlog_diff is best), monthly_min level, monthly_avg zscore**
- **Raw vs SymLog**: Monthly_max_diff has kurt=34 - **SYMLOG STRONGLY PREFERRED** (kurt drops to 5.3). Monthly_min is fine raw.

#### Financial Stress Index - MEDIUM
- **What it is**: St. Louis Fed Financial Stress Index
- **Why it matters**: Composite stress measure
- **Best transforms**: `_diff_zscore_12m` (r=0.155), only z-score transforms available
- **Recommendation**: **KEEP _diff_zscore_12m, CONDITIONAL on _diff_zscore_3m**
- **Raw vs SymLog**: Z-scores already normalized, raw is fine

### 2. Unifier (131 series: 92 KEEP, 29 CONDITIONAL, 10 DROP)

#### AHE_Private (Average Hourly Earnings) - MEDIUM but HIGH correlation
- **What it is**: Average hourly earnings for private sector workers
- **Why it matters**: Wage pressure indicator. NEGATIVE correlation means higher wages -> harder to hire -> fewer new jobs (cost pressure)
- **r = -0.634 (NSA), -0.780 (SA)** - second highest NSA, HIGHEST SA correlation
- **Problem**: Massively leptokurtic - skew=5.7, kurtosis=62 for _diff
- **Best transforms**: `_diff` (r=0.634), `_pct_chg` (r=0.618), `_symlog_diff` (r=0.615)
- **Recommendation**: **KEEP _diff + _symlog_diff + _diff_zscore_12m. DROP level (r=0.038) and _zscore_3m variants (r=0.077)**
- **Raw vs SymLog**: **SYMLOG ESSENTIAL** for _diff and _pct_chg. Z-score versions are fine raw.

#### AWH_All_Private (Average Weekly Hours) - CRITICAL
- **What it is**: Average weekly hours worked
- **Why it matters**: **HOURS ARE CUT BEFORE HEADCOUNT**. Most important leading indicator.
- **Max correlation**: 0.175 (NSA), 0.214 (SA) - moderate but fundamentally critical
- **Recommendation**: **KEEP level + diff + pct_chg** regardless of correlation
- **Raw vs SymLog**: Well-behaved distribution, raw is fine

#### AWH_Manufacturing - HIGH
- **What it is**: Manufacturing average weekly hours
- **Why it matters**: Manufacturing hours lead manufacturing employment
- **Max correlation**: 0.211 (NSA), 0.289 (SA)
- **Recommendation**: **KEEP level + _diff_zscore_3m**
- **Raw vs SymLog**: Level has VIF=4.2 (good!), z-scores are massively collinear. Raw is fine for level.

#### ISM Manufacturing/NonManufacturing PMI - HIGH
- **What it is**: Purchasing Managers' Index surveys
- **Why it matters**: Leading economic indicator, employment sub-component
- **ISM Mfg best**: `_symlog_pct_chg` (r=0.267 NSA, 0.396 SA)
- **ISM NonMfg best**: level (r=0.227 NSA, 0.315 SA)
- **Recommendation**: **KEEP level + symlog_pct_chg for both. DROP redundant z-score transforms**
- **Raw vs SymLog**: ISM level is well-behaved. _pct_chg has kurt=8.2 - symlog preferred

#### Challenger Job Cuts - HIGH
- **What it is**: Challenger, Gray & Christmas layoff announcements
- **Why it matters**: Direct layoff signal. Negative = more layoffs = fewer NFP gains
- **r = -0.531 (NSA), -0.587 (SA)** for _diff - very strong
- **Problem**: Very skewed (4.8) and leptokurtic (40.4) at level
- **Recommendation**: **KEEP _diff + _diff_zscore_12m. Symlog preferred for _diff**
- **Raw vs SymLog**: **SYMLOG ESSENTIAL** for level and _diff (both have extreme kurtosis)

#### CB Consumer Confidence - MEDIUM
- **What it is**: Conference Board Consumer Confidence Index
- **Why it matters**: Consumer sentiment -> spending -> service sector hiring
- **Best**: `_diff` (r=0.218 NSA, 0.284 SA)
- **Recommendation**: **KEEP _diff + _diff_zscore_3m**
- **Raw vs SymLog**: Raw is fine

#### Housing Starts - MEDIUM
- **What it is**: New residential construction starts
- **Why it matters**: Construction employment, wealth effect
- **Best**: `_symlog_diff` (r=0.173 NSA, 0.218 SA)
- **Recommendation**: **KEEP _symlog_diff + _diff_zscore_12m**
- **Raw vs SymLog**: Level is fine raw (VIF=1.95, lowest in the dataset!). But _diff benefits from symlog.

#### Retail Sales - MEDIUM
- **What it is**: Monthly retail and food services sales
- **Why it matters**: Consumer demand -> retail/service hiring
- **Best**: `_symlog_pct_chg_zscore_12m` (r=0.147 NSA, 0.160 SA)
- **Problem**: EXTREMELY leptokurtic at level (kurt=97.8, skew=10.0) and _pct_chg (kurt=200!)
- **Recommendation**: **KEEP _symlog_pct_chg_zscore_12m ONLY. DROP level and raw _pct_chg**
- **Raw vs SymLog**: **SYMLOG ABSOLUTELY ESSENTIAL**. Raw _pct_chg is unusable (kurt=200).

#### Industrial Production - MEDIUM
- **What it is**: Total industrial output index
- **Why it matters**: Factory output -> manufacturing jobs
- **r = 0.559 (NSA), 0.753 (SA)** for _symlog_diff - top 5 predictor for both targets
- **Problem**: _diff has kurt=170.7, skew=-7.6 (extreme negative tail from COVID)
- **Recommendation**: **KEEP _symlog_diff + _symlog_pct_chg + _diff_zscore_12m**
- **Raw vs SymLog**: **SYMLOG ABSOLUTELY ESSENTIAL**. Raw _diff is nearly unusable.

#### Empire State Manufacturing - LOW
- **What it is**: NY Fed manufacturing survey
- **Why it matters**: Regional PMI, noisy but timely
- **Best**: Level (r=0.287 NSA, 0.322 SA) - actually decent
- **Recommendation**: **KEEP level ONLY. DROP all transforms** (_pct_chg has kurt=98, _symlog_pct_chg kurt=41)
- **Raw vs SymLog**: Level is well-behaved (skew=-0.8, kurt=2.3). **RAW ONLY for level.**

#### UMich Consumer Expectations - LOW
- **What it is**: University of Michigan consumer expectations survey
- **Why it matters**: Forward-looking sentiment
- **Max correlation**: 0.035 (NSA), 0.080 (SA) - very weak
- **Recommendation**: **CONDITIONAL** - keep level only if ablation study shows value
- **Raw vs SymLog**: Raw level is perfect (kurt=1.87). Z-scores are also fine raw.

**DROP RECOMMENDATIONS from Unifier** (10 series):
- `Empire_State_Mfg_pct_chg` (kurt=98, r=0.032)
- `Empire_State_Mfg_symlog_pct_chg` (kurt=41, r=0.012)
- `Empire_State_Mfg_symlog_diff` (r=0.057)
- `Empire_State_Mfg_diff` (r=0.038)
- `AHE_Private_diff_zscore_3m` (N/A correlation - data issue)
- Various weak transforms with insufficient observations

### 3. Prosper (185 series: 38 KEEP, 143 CONDITIONAL, 4 DROP)

**The most bloated source.** 185 original series from only ~5 survey questions across 4 demographic groups, each with 4-9 transforms. Massive redundancy.

#### Key Finding: Only Females demographic shows strong signal
The "I know people who have been laid off | Females" series is the 7th strongest predictor overall (r=-0.553 NSA, -0.704 SA). But the Males, 18-34, and US 18+ variants of the same question are much weaker.

#### Employment Environment: "I know people laid off" - HIGH value for Females only
- Females _pct_chg: r=0.553 NSA, 0.704 SA (**KEEP, SYMLOG PREFERRED**)
- Females _diff: r=0.297/0.325 (**KEEP, SYMLOG PREFERRED**)
- Males best: r=0.149/0.224 (KEEP marginally)
- 18-34 best: r=0.115/0.128 (CONDITIONAL)

#### Layoff Outlook: "More/Same/Fewer layoffs expected"
- "More" Females: r=0.301/0.519 (**KEEP**)
- "Same" Females: r=0.144/0.241 (KEEP)
- "Fewer" anything: r < 0.06 (**DROP** - no signal)
- Most 18-34 and Males variants: r < 0.10 (CONDITIONAL to DROP)

#### Consumer Spending Forecast
- Males _diff_zscore_12m: r=0.388 (**KEEP** - surprising outlier)
- 18-34 level: r=0.269 (KEEP)
- Females: r=0.195 (CONDITIONAL)

#### Consumer Mood Index
- Males level: r=0.257 (KEEP)
- Females level: r=0.215 (KEEP)
- 18-34: weak (r=0.129), mostly CONDITIONAL

#### Employment: "I am employed" / "I am unemployed"
- Most variants: r < 0.15 (CONDITIONAL)
- "I am employed full-time" 18-34 _diff: r=0.163/0.218 (KEEP)
- "I am unemployed" Males _pct_chg: r=0.239 (KEEP)

**BRUTAL RECOMMENDATION for Prosper**: Of 185 original series, only ~38 are KEEP. The remaining 143 CONDITIONAL should be tested via ablation - most will be noise. The 4 DROPs are data-quality issues (too few observations).

**Key insight**: Prosper survey data is useful primarily through **Females** demographic breakdowns and the **"I know people who have been laid off"** question. The other questions and demographics add limited value.

### 4. ADP (1 series: 1 KEEP)

#### ADP_actual - CRITICAL
- **What it is**: ADP private payroll estimate (released 2 days before NFP)
- **Why it matters**: Most direct NFP predictor available. Same methodology, private sector only.
- **r = 0.608 (NSA), 0.762 (SA)** - top 5 for both targets
- **Problem**: EXTREMELY skewed (skew=-8.6, kurt=90.5) from COVID
- **Recommendation**: **KEEP. SYMLOG ESSENTIAL** (reduces kurt from 90.5 to 7.3)
- Only 191 observations (since 2010) but fundamentally irreplaceable

### 5. NOAA (2 series: 1 KEEP, 1 CONDITIONAL)

#### NOAA_Human_Impact_Index - CONDITIONAL
- 430 observations, long history
- Very low correlation: r=0.027 NSA, 0.037 SA
- Highly skewed (weather disasters are rare events)
- **Recommendation**: **CONDITIONAL** - may capture extreme weather events but signal is very weak

#### NOAA_Economic_Damage_Index - KEEP (barely)
- Similar to Human Impact but captures economic costs
- r=0.078 NSA - slightly stronger
- **Recommendation**: **KEEP** but with low confidence; symlog recommended

---

## Distributional Analysis: Most Problematic Series

### Top 10 Highest Kurtosis (most extreme tails)

| Series | Kurtosis | Skewness | Cause | Fix |
|--------|----------|----------|-------|-----|
| Retail_Sales_pct_chg | **200.1** | 14.2 | COVID spike | SymLog or DROP |
| Industrial_Production_diff | **170.7** | -7.6 | COVID crash | SymLog |
| Oil_Prices_volatility_diff | **136.4** | -2.4 | Oil crises | SymLog |
| Retail_Sales_diff | **133.5** | 3.8 | COVID spike | SymLog |
| Oil_Prices_volatility | **107.3** | 7.9 | Oil crises | SymLog |
| Empire_State_Mfg_pct_chg | **98.1** | -8.9 | Volatile | DROP |
| Retail_Sales (level) | **97.8** | 10.0 | Trend + spike | SymLog z-score |
| ADP_actual | **90.5** | -8.6 | COVID | SymLog |
| Weekly_Econ_Index_max_pct_chg | **77.3** | 4.0 | COVID | SymLog |
| "I know laid off" Females_pct_chg | **69.8** | 5.9 | COVID | SymLog |

### Series Where SymLog Reduces Kurtosis the Most

| Series | Raw Kurtosis | SymLog Kurtosis | Improvement |
|--------|-------------|-----------------|-------------|
| Industrial_Production_diff | 170.7 | 6.7 | **164.0** |
| Retail_Sales_symlog_pct_chg | 199.8 | 38.2 | **161.6** |
| Oil_Prices_volatility_diff | 136.4 | 13.5 | **122.9** |
| Oil_Prices_volatility | 107.3 | 3.9 | **103.5** |
| ADP_actual | 90.5 | 7.3 | **83.2** |

---

## VIF Analysis: Multicollinearity

**Critical finding**: Because the ETL pipeline generates multiple transforms per base series (level, diff, pct_chg, zscore_3m, zscore_12m, symlog variants), VIF is astronomical. Every transform of the same base indicator is massively collinear with the others.

### Lowest VIF (most independent features)

| Series | VIF | Interpretation |
|--------|-----|---------------|
| UMich_Expectations (level) | 1.87 | Very independent |
| Housing_Starts (level) | 1.95 | Very independent |
| ISM_Manufacturing_Index (level) | 2.41 | Independent |
| Industrial_Production (level) | 2.54 | Independent |
| AWH_Manufacturing (level) | 4.23 | Acceptable |

**Key insight**: Level values have low VIF. All derived transforms (diff, zscore, etc.) have extreme VIF because they're computed from the same base data. This means: **for each base indicator, keep at most 2-3 transforms (the best-correlated ones) and drop the rest**.

---

## Raw vs SymLog Recommendations

### SYMLOG ESSENTIAL (raw is nearly unusable)
- ADP_actual (kurt 90 -> 7)
- Industrial_Production_diff (kurt 171 -> 7)
- Oil_Prices_volatility (kurt 107 -> 4)
- Oil_Prices_volatility_diff (kurt 136 -> 14)
- Retail_Sales_pct_chg (kurt 200 -> 38)
- Challenger_Job_Cuts level (kurt 40 -> 1)
- AHE_Private_diff (kurt 62 -> 8)

### SYMLOG PREFERRED (improves but raw is usable)
- Weekly_Econ_Index_monthly_max_diff (kurt 34 -> 5)
- Weekly_Econ_Index_monthly_max_pct_chg (kurt 77 -> 1)
- CCSA_monthly_avg (kurt 49 -> 7)
- ISM_Manufacturing_Index_pct_chg (kurt 8 -> 2)
- Various Prosper survey _pct_chg transforms

### RAW ONLY (symlog adds nothing or hurts)
- All z-score transforms (_zscore_3m, _zscore_12m) - already normalized
- Binary flags (VIX_panic_regime, SP500_crash_month, etc.)
- Level values with low kurtosis (ISM, AWH, Housing_Starts, UMich, Yield_Curve)
- CCNSA/CCSA z-score transforms (already well-behaved)
- Empire_State_Mfg level (skew=-0.8, kurt=2.3 - near-normal)

### BOTH (keep raw and symlog, different signal)
- Industrial_Production_symlog_diff + Industrial_Production_diff_zscore_12m
- Credit_Spreads_avg (level raw + acceleration symlog)
- NOAA indices (rare events benefit from both perspectives)

---

## Final Actionable Recommendations

### IMMEDIATE ACTIONS (high impact, no risk)

1. **DROP these 14 series entirely**:
   - Empire_State_Mfg: _pct_chg, _symlog_pct_chg, _symlog_diff, _diff (but KEEP level)
   - AHE_Private_diff_zscore_3m (data quality issue)
   - Various Prosper series with < 24 observations
   - Empire_State_Mfg_pct_chg_zscore_12m, _symlog counterparts

2. **Convert to SymLog for these 61 series** where raw is problematic:
   - All ADP, Industrial_Production diffs, Oil_Prices volatility, Challenger_Job_Cuts diffs, AHE_Private diffs, Retail_Sales pct_chg, etc.

### RECOMMENDED FEATURE REDUCTION (transform deduplication)

For each base indicator, keep only the **best 2-3 transforms** based on correlation and distribution:

| Base Indicator | Keep Transforms | Drop Transforms |
|----------------|----------------|-----------------|
| CCNSA_monthly_avg | _diff_zscore_12m, _pct_chg_zscore_12m | symlog variants (redundant r=0.626 vs 0.643) |
| AHE_Private | _symlog_diff, _diff_zscore_12m | _pct_chg (redundant), all _zscore_3m |
| ADP_actual | symlog level | (only 1 series) |
| Weekly_Econ_Index_max | _symlog_diff, _symlog_pct_chg | _pct_chg (raw kurt=77), _zscore_3m |
| Industrial_Production | _symlog_diff, _symlog_pct_chg, _diff_zscore_12m | raw _diff (kurt=171), all _zscore_3m |
| Challenger_Job_Cuts | _diff (symlog), _diff_zscore_12m | level (r=0.437 but kurt=40), _zscore_3m |
| Oil_Prices_volatility | level (symlog), _diff (symlog) | all z-score transforms (weak r) |
| ISM_Manufacturing | level, _symlog_pct_chg | _diff, all z-scores |
| Empire_State_Mfg | level ONLY | ALL transforms |
| Prosper "Laid off" Females | _pct_chg (symlog), _diff (symlog) | _zscore_3m, _zscore_12m |

### PROSPER CLEANUP (biggest reduction opportunity)

**Current**: 185 series
**Recommended**: ~30-40 series (70-80% reduction)

Keep:
- "I know people laid off" | Females: _pct_chg + _symlog_diff (top predictors)
- "More layoffs" | Females: _pct_chg + _diff
- "Same layoffs" | Females: level
- Consumer Spending Forecast | Males: _diff_zscore_12m
- Consumer Spending Forecast | 18-34: level
- Consumer Mood Index | Males: level
- Consumer Mood Index | Females: level + _diff
- "I am employed full-time" | 18-34: _diff
- "I am unemployed" | Males: _pct_chg
- "I am unemployed" | 18-34: level

Drop everything else from Prosper (all "Fewer layoffs" variants, most demographic×transform combinations, all redundant symlog z-score pairs).

### ESTIMATED FEATURE COUNTS AFTER CLEANUP

| Phase | Count | Description |
|-------|-------|-------------|
| Current | 553 original + 553 symlog = 1,106 | Everything |
| After DROP | 539 + 539 = 1,078 | Remove 14 DROPs |
| After transform dedup | ~150 original + ~60 symlog = ~210 | Best transforms only |
| After Prosper cleanup | ~120 original + ~50 symlog = ~170 | Core features |

This ~170 feature set would be dramatically more efficient than the current ~1,100 while retaining all high-correlation signals.
