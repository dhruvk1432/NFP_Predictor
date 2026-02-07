"""
Data_ETA_Pipeline - Consolidated Data Loading and Preparation
=============================================================

Merges the former Load_Data/ and Prepare_Data/ folders into unified pipelines.

Key pipeline files:
- fred_employment_pipeline.py: FRED employment data + NFP targets + BLS schedule + preparation
- noaa_pipeline.py: NOAA storm events → state-level → master → NFP-weighted snapshots
- adp_pipeline.py: ADP employment CSV → MoM changes → NFP-aligned snapshots

Shared utilities re-exported here for convenience:
- NFP relative timing functions (used by exogenous data loaders)
"""

# Re-export NFP timing utilities so other files can import from the package directly.
# These live in fred_employment_pipeline.py but are used by load_fred_exogenous,
# load_prosper_data, load_unifier_data, and noaa_pipeline.
from Data_ETA_Pipeline.fred_employment_pipeline import (
    load_nfp_releases,
    get_nfp_release_for_month,
    get_nfp_release_map,
    calculate_median_offset_from_nfp,
    apply_nfp_relative_adjustment,
    get_series_timing_stats,
)
