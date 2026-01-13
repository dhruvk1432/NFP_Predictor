# Load_Data Module

Data ingestion layer for the NFP Predictor system. This module fetches raw data from external sources and saves it in a point-in-time snapshot format organized by decade/year/month.

## Overview

The Load_Data module is responsible for:
- Fetching raw data from multiple external APIs and web sources
- Organizing data into monthly snapshots to preserve point-in-time correctness
- Handling API rate limits, retries, and error recovery
- Supporting both real-time updates and historical backfills

## Data Sources

| Source | Data Type | Update Frequency | API/Method |
|--------|-----------|------------------|------------|
| FRED | Employment levels (NSA/SA), Financial indicators | Monthly | REST API |
| Unifier | ISM, Consumer Confidence, JOLTS | Monthly | REST API |
| ADP | Employment forecast/actual | Monthly | Web scraping |
| NOAA | Storm events, damage data | Monthly | REST API |
| Prosper | Consumer sentiment surveys | Monthly | REST API |
| BLS | NFP release schedule | Annual | Web scraping |

## Directory Structure

```
Load_Data/
├── __init__.py
├── README.md                      # This file
├── utils.py                       # Shared utilities
├── fred_snapshots.py              # FRED employment data
├── load_fred_exogenous.py         # FRED financial indicators
├── load_unifier_data.py           # ISM, Confidence, JOLTS
├── load_ADP_Employment_change.py  # ADP employment data
├── load_noaa_data.py              # Storm event data
├── load_prosper_data.py           # Consumer surveys
└── scrape_bls_schedule.py         # NFP release dates
```

## File Descriptions

### `fred_snapshots.py`
**Purpose:** Downloads NFP employment data (NSA and SA versions) from FRED API.

**Key Features:**
- Fetches 80+ employment series across all BLS hierarchy levels
- Supports both NSA (non-seasonally adjusted) and SA (seasonally adjusted) data
- Handles FRED API rate limiting (429 errors) with exponential backoff
- Organizes output by decade/year/month structure

**Output:** `data/fred_data/decades/{decade}/{year}/{YYYY-MM}.parquet`

**Usage:**
```bash
# Fetch all historical data
python Load_Data/fred_snapshots.py

# Fetch specific date range
python Load_Data/fred_snapshots.py --start 2020-01-01 --end 2024-12-31
```

**Key Series Fetched:**
- Total Nonfarm (PAYEMS/PAYNSA)
- Private vs Government breakdown
- Goods-producing vs Service-providing
- 40+ sector-level employment series

---

### `load_fred_exogenous.py`
**Purpose:** Downloads exogenous financial indicators from FRED API.

**Key Features:**
- Fetches VIX, S&P 500, Oil prices, Credit spreads, Yield curve data
- Calculates derived features (Z-scores, crash indicators, volatility)
- Creates panic regime flags (VIX > 50, circuit breakers)

**Output:** `data/Exogenous_data/exogenous_fred_data/decades/{decade}/{year}/{YYYY-MM}.parquet`

**Indicators Fetched:**
| Category | Series | Description |
|----------|--------|-------------|
| Volatility | VIX | CBOE Volatility Index |
| Equity | S&P 500 | Stock market returns, drawdowns |
| Credit | BAA-AAA Spread | Credit risk premium |
| Rates | 10Y-3M Spread | Yield curve slope |
| Energy | WTI Oil | Crude oil prices |
| Stress | STLFSI | St. Louis Financial Stress Index |
| Labor | ICSA, CCSA | Weekly jobless claims |

**Derived Features:**
- `VIX_panic_regime`: Binary flag when VIX > 50
- `SP500_crash_month`: Monthly return < -10%
- `SP500_circuit_breaker`: Any day down > 5%
- Rolling Z-scores for extreme event detection

---

### `load_unifier_data.py`
**Purpose:** Fetches ISM Manufacturing, Consumer Confidence, and JOLTS data from Unifier API.

**Key Features:**
- ISM Manufacturing Employment Index (leading indicator)
- Conference Board Consumer Confidence
- JOLTS Job Openings and Layoffs data
- Aligns data to NFP release timing

**Output:** `data/Exogenous_data/exogenous_unifier_data/decades/{decade}/{year}/{YYYY-MM}.parquet`

**Environment Variables Required:**
```env
UNIFIER_USER=your_username
UNIFIER_TOKEN=your_api_token
```

---

### `load_ADP_Employment_change.py`
**Purpose:** Scrapes ADP National Employment Report data from investing.com.

**Key Features:**
- ADP report releases 2 days before official NFP
- Captures both forecast and actual values
- Uses Selenium for web scraping
- Handles dynamic page loading

**Output:** `data/Exogenous_data/ADP_data/{YYYY-MM}.parquet`

**Data Fields:**
- `ADP_actual`: Reported employment change
- `ADP_forecast`: Consensus forecast
- `ADP_surprise`: Actual minus forecast

**Note:** Requires Chrome WebDriver for Selenium.

---

### `load_noaa_data.py`
**Purpose:** Downloads storm event data from NOAA Storm Events Database.

**Key Features:**
- State-level storm event aggregation
- Property and crop damage amounts
- Deaths and injuries (direct/indirect)
- Inflation adjustment to 2020 dollars

**Output:** `data/Exogenous_data/NOAA_data/`

**Data Fields:**
- Deaths (direct/indirect)
- Injuries (direct/indirect)
- Property damage (inflation-adjusted)
- Crop damage (inflation-adjusted)

---

### `load_prosper_data.py`
**Purpose:** Fetches consumer sentiment survey data from Prosper Insights.

**Key Features:**
- Consumer financial sentiment indicators
- Spending intention surveys
- Employment outlook measures

**Output:** `data/Exogenous_data/prosper/decades/{decade}/{year}/{YYYY-MM}.parquet`

---

### `scrape_bls_schedule.py`
**Purpose:** Retrieves future NFP release dates from BLS website.

**Key Features:**
- Parses BLS release calendar
- Used to align other data sources to NFP timing
- Caches release dates for efficiency

**Usage:**
```python
from Load_Data.scrape_bls_schedule import get_future_nfp_dates

# Get next 12 release dates
dates = get_future_nfp_dates(n_months=12)
```

---

### `utils.py`
**Purpose:** Shared utility functions for data loading modules.

**Key Functions:**
```python
def get_snapshot_path(base_dir, date, create=True):
    """Build decade/year/month snapshot path."""

def flatten_multiindex_columns(df):
    """Flatten MultiIndex columns from pivot operations."""
```

## Output Data Format

All modules output data in **long format** with consistent columns:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Data observation date |
| `series_name` | str | Identifier for the data series |
| `value` | float | The data value |
| `snapshot_date` | datetime | When this snapshot was created |
| `release_date` | datetime | When this data was released (optional) |

## Point-in-Time Correctness

Data is organized to ensure **no look-ahead bias**:

1. Each snapshot represents data available as of a specific NFP release date
2. Data is not revised after initial snapshot creation
3. Historical backfills recreate point-in-time views

```
data/fred_data/decades/
├── 2010s/
│   ├── 2019/
│   │   ├── 2019-01.parquet  # Data available for Jan 2019 NFP
│   │   ├── 2019-02.parquet  # Data available for Feb 2019 NFP
│   │   └── ...
│   └── ...
├── 2020s/
│   ├── 2020/
│   │   ├── 2020-01.parquet
│   │   ├── 2020-02.parquet
│   │   └── ...
│   └── ...
```

## Error Handling

All modules implement robust error handling:

- **API Rate Limiting:** Exponential backoff with jitter
- **Network Errors:** Automatic retry with configurable attempts
- **Missing Data:** Graceful handling with logging
- **Validation:** Data integrity checks before saving

## Environment Configuration

Required environment variables (set in `.env`):

```env
# Required
FRED_API_KEY=your_fred_api_key

# Optional (for specific modules)
UNIFIER_USER=your_username
UNIFIER_TOKEN=your_token
DATA_PATH=./data
START_DATE=1990-01-01
END_DATE=2025-12-31
```

## Running Data Loads

### Full Historical Load
```bash
# Load all data sources
python Load_Data/fred_snapshots.py
python Load_Data/load_fred_exogenous.py
python Load_Data/load_unifier_data.py
python Load_Data/load_noaa_data.py
python Load_Data/load_prosper_data.py
```

### Incremental Update
```bash
# Update only recent months
python Load_Data/fred_snapshots.py --start 2024-01-01
python Load_Data/load_fred_exogenous.py --start 2024-01-01
```

### Using run_full_project.py
```bash
# Run all load steps
python run_full_project.py --stage load

# Skip specific data sources
python run_full_project.py --stage load --skip noaa,prosper
```

## Dependencies

- `fredapi`: FRED API client
- `requests`: HTTP requests
- `selenium`: Web scraping (for ADP)
- `pandas`: Data manipulation
- `numpy`: Numerical operations

## Related Modules

- **Prepare_Data:** Transforms raw data into features
- **Train:** Uses prepared data for model training
- **utils/paths.py:** Centralized path generation
