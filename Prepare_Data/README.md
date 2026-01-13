# Prepare_Data Module

Data transformation and consolidation layer for the NFP Predictor system. This module transforms raw data into features and consolidates all sources into master snapshots ready for model training.

## Overview

The Prepare_Data module is responsible for:
- Transforming raw employment data (levels to MoM changes)
- Applying mathematical transforms to handle extreme values (SymLog, Log1p)
- Aggregating NOAA storm data with employment weights
- Consolidating all data sources into unified master snapshots
- Ensuring point-in-time correctness is preserved

## Data Pipeline Position

```
Load_Data (raw) → Prepare_Data (features) → Train (model)
```

## Directory Structure

```
Prepare_Data/
├── __init__.py
├── README.md                        # This file
├── create_master_snapshots.py       # Main consolidation script
├── prepare_fred_snapshots.py        # FRED data preprocessing
├── create_adp_snapshots.py          # ADP data alignment
├── create_noaa_master.py            # NOAA state-level aggregation
├── create_noaa_weighted.py          # NOAA employment-weighted aggregation
└── analyze_noaa_weights_vintage.py  # NOAA weight analysis utility
```

## File Descriptions

### `create_master_snapshots.py`
**Purpose:** Consolidates all exogenous data sources into unified master snapshots.

**Key Features:**
- Merges FRED exogenous, Unifier, ADP, NOAA, and Prosper data
- Applies data transformations (SymLog, Log1p, percent change)
- Creates dimension-reduced NOAA features
- Maintains point-in-time snapshot structure

**Output:** `data/Exogenous_data/master_snapshots/decades/{decade}/{year}/{YYYY-MM}.parquet`

**Usage:**
```bash
# Create all master snapshots
python Prepare_Data/create_master_snapshots.py

# Create for specific date range
python Prepare_Data/create_master_snapshots.py --start 2020-01-01 --end 2024-12-31
```

**Data Sources Merged:**
| Source | Input Path | Key Series |
|--------|------------|------------|
| FRED Exogenous | `exogenous_fred_data/` | VIX, S&P 500, Oil, Credit Spreads |
| Unifier | `exogenous_unifier_data/` | ISM, Consumer Confidence, JOLTS |
| ADP | `ADP_snapshots/` | ADP actual, forecast, surprise |
| NOAA | `noaa_weighted_snapshots/` | Storm damage, deaths, injuries |
| Prosper | `prosper/` | Consumer sentiment |

**Transformations Applied:**

1. **Percent Change (Stationarity Fix):**
   - `CCSA_monthly_avg` → `CCSA_monthly_avg_pct_change`
   - Converts levels to growth rates for stationarity

2. **SymLog Transform (Crash Handling):**
   - Formula: `sign(x) * log1p(abs(x))`
   - Applied to: ADP_actual, Credit_Spreads changes, Oil crashes
   - Handles negative values and compresses outliers

3. **Log1p Transform (Skew Reduction):**
   - Formula: `log(1 + x)`
   - Applied to: JOLTS_Layoffs, VIX metrics, volatility measures
   - Reduces positive skewness

4. **NOAA Aggregation:**
   - Human impact: deaths + injuries → single composite
   - Economic damage: property + crop damage → single composite

---

### `prepare_fred_snapshots.py`
**Purpose:** Preprocesses FRED employment data for model consumption.

**Key Features:**
- Converts employment levels to Month-over-Month (MoM) changes
- Applies SymLog transform to handle extreme values (COVID crash)
- Applies RobustScaler for normalization
- Preserves point-in-time correctness

**Output:** `data/fred_data_prepared/decades/{decade}/{year}/{YYYY-MM}.parquet`

**Transformation Pipeline:**
```
Raw Employment Levels
        ↓
Month-over-Month Change (diff)
        ↓
SymLog Transform: sign(x) * log1p(|x|)
        ↓
RobustScaler (median/IQR normalization)
        ↓
Prepared Employment Features
```

**Why SymLog?**
The NFP MoM changes have extreme outliers (COVID: -20 million jobs). SymLog:
- Reduces skewness: -6.5 → -1.1
- Reduces kurtosis: 81 → -0.7
- Preserves sign and relative magnitude
- Is fully invertible for prediction recovery

**Usage:**
```bash
python Prepare_Data/prepare_fred_snapshots.py
```

---

### `create_adp_snapshots.py`
**Purpose:** Aligns ADP employment data with NFP release timing.

**Key Features:**
- Aligns ADP release dates (2 days before NFP) to NFP months
- Calculates surprise (actual - forecast)
- Handles missing data and outliers

**Output:** `data/Exogenous_data/ADP_snapshots/decades/{decade}/{year}/{YYYY-MM}.parquet`

**Features Created:**
| Feature | Description |
|---------|-------------|
| `ADP_actual` | Reported employment change (thousands) |
| `ADP_forecast` | Consensus forecast |
| `ADP_surprise` | Actual minus forecast |
| `ADP_surprise_pct` | Surprise as percentage of forecast |

---

### `create_noaa_master.py`
**Purpose:** Aggregates raw NOAA storm events to state level.

**Key Features:**
- Aggregates individual storm events by state and month
- Sums deaths, injuries, property damage, crop damage
- Handles inflation adjustment to 2020 dollars

**Output:** State-level NOAA summaries

**Aggregation:**
```
Individual Storm Events
        ↓
Group by State + Month
        ↓
Sum: deaths, injuries, damage
        ↓
State-Level Summaries
```

---

### `create_noaa_weighted.py`
**Purpose:** Creates employment-weighted national NOAA aggregates.

**Key Features:**
- Weights state-level NOAA data by employment share
- States with more employees contribute more to national totals
- Log transforms damage amounts to reduce skewness

**Output:** `data/Exogenous_data/noaa_weighted_snapshots/decades/{decade}/{year}/{YYYY-MM}.parquet`

**Weighting Logic:**
```
National Value = Σ (State Value × State Employment Share)

where State Employment Share = State Employment / Total US Employment
```

**Features Created:**
| Feature | Description |
|---------|-------------|
| `deaths_direct_weighted_log` | Log of weighted direct deaths |
| `deaths_indirect_weighted_log` | Log of weighted indirect deaths |
| `injuries_direct_weighted_log` | Log of weighted direct injuries |
| `injuries_indirect_weighted_log` | Log of weighted indirect injuries |
| `total_property_damage_real_weighted_log` | Log of inflation-adjusted property damage |
| `total_crop_damage_real_weighted_log` | Log of inflation-adjusted crop damage |
| `noaa_human_impact` | Composite: deaths + injuries |
| `noaa_economic_damage` | Composite: property + crop damage |

---

### `analyze_noaa_weights_vintage.py`
**Purpose:** Utility script for analyzing NOAA employment weights over time.

**Key Features:**
- Analyzes how employment weights change over time
- Validates weight calculations
- Diagnostic tool for debugging

## Master Snapshot Format

The master snapshot contains all exogenous features in **long format**:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Data observation date |
| `series_name` | str | Feature identifier |
| `value` | float | The feature value |
| `snapshot_date` | datetime | NFP release month |
| `release_date` | datetime | When underlying data was released |

**Example Series Names:**
```
VIX_max
VIX_mean
VIX_panic_regime
SP500_monthly_return
SP500_crash_month
Credit_Spreads_avg
Oil_Prices_mean
CCSA_monthly_avg
ISM_Manufacturing
ADP_actual
noaa_human_impact
noaa_economic_damage
```

## Transformation Reference

### SymLog Transform
```python
def apply_symlog(x):
    """sign(x) * log1p(|x|)"""
    return np.sign(x) * np.log1p(np.abs(x))

def inverse_symlog(y):
    """Inverse for prediction recovery"""
    return np.sign(y) * (np.exp(np.abs(y)) - 1)
```

**Applied to:**
- `ADP_actual`
- `Credit_Spreads_monthly_chg`
- `Yield_Curve_monthly_chg`
- `Oil_Prices_30d_crash`
- `Credit_Spreads_zscore_max`

### Log1p Transform
```python
def apply_log1p(x):
    """log(1 + x) for positive values"""
    return np.log1p(x)
```

**Applied to:**
- `JOLTS_Layoffs`
- `Challenger_Job_Cuts`
- `Oil_Prices_volatility`
- `VIX_mean`, `VIX_max`, `VIX_volatility`
- `SP500_volatility`

### Series NOT Transformed
These are preserved raw for linear extrapolation during extreme events:
- `CCSA_MoM_Pct` (claims spike magnitude)
- `SP500_monthly_return` (crash magnitude)
- `SP500_30d_return`
- `SP500_max_drawdown`

## Running Preparation Steps

### Full Pipeline
```bash
# Prepare all data
python Prepare_Data/prepare_fred_snapshots.py
python Prepare_Data/create_noaa_master.py
python Prepare_Data/create_noaa_weighted.py
python Prepare_Data/create_adp_snapshots.py
python Prepare_Data/create_master_snapshots.py
```

### Using run_full_project.py
```bash
# Run all prepare steps
python run_full_project.py --stage prepare
```

### Incremental Update
```bash
# Only update recent months
python Prepare_Data/create_master_snapshots.py --start 2024-01-01
```

## Output Directory Structure

```
data/
├── fred_data/                    # Raw FRED employment (from Load_Data)
│   └── decades/
├── fred_data_prepared/           # Transformed employment data
│   └── decades/
│       └── 2020s/
│           └── 2024/
│               ├── 2024-01.parquet
│               ├── 2024-02.parquet
│               └── ...
└── Exogenous_data/
    ├── master_snapshots/         # Consolidated features
    │   └── decades/
    │       └── 2020s/
    │           └── 2024/
    │               ├── 2024-01.parquet
    │               └── ...
    ├── ADP_snapshots/            # Aligned ADP data
    └── noaa_weighted_snapshots/  # Weighted NOAA data
```

## Data Quality Checks

The module performs several validation steps:
- **Missing Values:** Logged and handled gracefully
- **Outlier Detection:** Extreme values flagged but preserved
- **Date Alignment:** Ensures all data aligns to NFP release months
- **Duplicate Detection:** Removes duplicate observations

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: RobustScaler for normalization
- `utils/transforms.py`: Shared transformation functions

## Related Modules

- **Load_Data:** Provides raw input data
- **Train:** Consumes master snapshots for model training
- **utils/transforms.py:** SymLog, Log1p, Z-score functions
