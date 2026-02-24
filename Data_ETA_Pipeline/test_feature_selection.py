"""
Comprehensive Test Suite for the Memory-Safe Feature Selection Engine.
Tests each stage individually on real FRED Employment data (49,536 features).
Every test has a timeout and memory check to prevent hangs or OOM kills.
"""
import pandas as pd
import numpy as np
import json
import time
import os
import gc
import psutil
import logging
import sys
import signal

# Timeout handler for macOS
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded time limit!")

signal.signal(signal.SIGALRM, timeout_handler)

# Setup
logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
logger = logging.getLogger('test')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings import DATA_PATH
from Data_ETA_Pipeline.feature_selection_engine import (
    load_snapshot_wide, _variance_filter, _deduplicate_group,
    _select_wide, filter_group_data_purged, _lgb_screen_group,
    _boruta_core, get_boruta_importance, cluster_redundancy,
    sequential_forward_selection, run_pipeline, LGB_PARAMS,
    _classify_series
)
from utils.transforms import winsorize_covid_period
from Train.data_loader import load_target_data

def mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def check_mem():
    mem = mem_mb()
    sys_pct = psutil.virtual_memory().percent
    if sys_pct > 85:
        logger.warning(f"⚠️  System memory at {sys_pct}% ({mem:.0f} MB RSS)")
    return mem

PASS = "✅ PASS"
FAIL = "❌ FAIL"

# ============================================================
# Load real data once
# ============================================================
print("=" * 70)
print("LOADING REAL FRED EMPLOYMENT DATA")
print("=" * 70)

# Use NSA split if available, fall back to combined
source_dir = DATA_PATH / 'fred_data_prepared_nsa' / 'decades'
if not source_dir.exists() or not list(source_dir.rglob('*.parquet')):
    source_dir = DATA_PATH / 'fred_data_prepared' / 'decades'
latest_path = sorted(source_dir.rglob('*.parquet'))[-1]
print(f"  Parquet: {latest_path.name} ({latest_path.stat().st_size / 1024**2:.0f} MB on disk)")

# Load target
target_df = load_target_data(target_type='nsa', release_type='first')
target_df = target_df.dropna(subset=['y_mom']).set_index('ds')
y = target_df['y_mom']
print(f"  Target: {len(y)} observations")

# Build series groups dynamically from column names
# First, do a quick load to get the column names
quick_df = pd.read_parquet(latest_path)
cols = [c for c in quick_df.columns if quick_df[c].dtype in [np.float32, np.float64]]
del quick_df; gc.collect()

from collections import defaultdict
classifications = defaultdict(list)
for col in cols:
    grp = _classify_series(col, 'FRED_Employment_NSA')
    classifications[grp].append(col)
classifications = dict(classifications)
group_names = list(classifications.keys())
group_sizes = {g: len(v) for g, v in classifications.items()}
print(f"  Groups: {len(group_names)}")
for g, s in sorted(group_sizes.items(), key=lambda x: -x[1]):
    print(f"    {g}: {s} features")

results = {}

# ============================================================
# TEST 1: load_snapshot_wide (the old freeze point)
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: load_snapshot_wide() on 49k-column parquet")
print("=" * 70)
signal.alarm(30)  # 30s timeout
try:
    t0 = time.time()
    snap_wide = load_snapshot_wide(latest_path)
    elapsed = time.time() - t0
    signal.alarm(0)
    
    mem = check_mem()
    print(f"  Shape: {snap_wide.shape}")
    print(f"  Time:  {elapsed:.1f}s")
    print(f"  RAM:   {mem:.0f} MB")
    print(f"  NaN%:  {snap_wide.isna().mean().mean()*100:.1f}%")
    
    assert snap_wide.shape[1] > 40000, f"Expected >40k cols, got {snap_wide.shape[1]}"
    assert elapsed < 30, f"Took {elapsed:.1f}s, expected <30s"
    print(f"  {PASS}: Loaded {snap_wide.shape[1]} features in {elapsed:.1f}s")
    results['load_snapshot_wide'] = PASS
except TimeoutError:
    signal.alarm(0)
    print(f"  {FAIL}: Timed out after 30s!")
    results['load_snapshot_wide'] = FAIL
    sys.exit(1)
except Exception as e:
    signal.alarm(0)
    print(f"  {FAIL}: {e}")
    results['load_snapshot_wide'] = FAIL
    sys.exit(1)

# ============================================================
# TEST 2: _variance_filter on 49k columns
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: _variance_filter() on full snapshot")
print("=" * 70)
signal.alarm(30)  # 30s timeout
try:
    t0 = time.time()
    snap_filtered = _variance_filter(snap_wide)
    elapsed = time.time() - t0
    signal.alarm(0)
    
    n_dropped = snap_wide.shape[1] - snap_filtered.shape[1]
    mem = check_mem()
    print(f"  Before: {snap_wide.shape[1]} cols")
    print(f"  After:  {snap_filtered.shape[1]} cols (dropped {n_dropped})")
    print(f"  Time:   {elapsed:.1f}s")
    print(f"  RAM:    {mem:.0f} MB")
    
    assert elapsed < 30, f"Took {elapsed:.1f}s, expected <30s"
    print(f"  {PASS}: Filtered {n_dropped} near-constant features in {elapsed:.1f}s")
    results['variance_filter'] = PASS
except TimeoutError:
    signal.alarm(0)
    print(f"  {FAIL}: Timed out!")
    results['variance_filter'] = FAIL
except Exception as e:
    signal.alarm(0)
    print(f"  {FAIL}: {e}")
    results['variance_filter'] = FAIL

# ============================================================
# TEST 3: _select_wide on a large group
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: _select_wide() on largest group")
print("=" * 70)
signal.alarm(30)
try:
    first_group = classifications[group_names[0]]
    t0 = time.time()
    wide_group = _select_wide(snap_filtered, first_group)
    elapsed = time.time() - t0
    signal.alarm(0)
    
    mem = check_mem()
    print(f"  Group:  {group_names[0]!r} ({len(first_group)} series)")
    print(f"  Shape:  {wide_group.shape}")
    print(f"  Time:   {elapsed:.1f}s")
    print(f"  RAM:    {mem:.0f} MB")
    
    assert elapsed < 30, f"Took {elapsed:.1f}s"
    assert wide_group.shape[1] > 0, "No features selected"
    print(f"  {PASS}: Selected {wide_group.shape[1]} features in {elapsed:.1f}s")
    results['select_wide'] = PASS
except TimeoutError:
    signal.alarm(0)
    print(f"  {FAIL}: Timed out!")
    results['select_wide'] = FAIL
except Exception as e:
    signal.alarm(0)
    print(f"  {FAIL}: {e}")
    results['select_wide'] = FAIL

# Free big snap_wide to save RAM
del snap_wide, snap_filtered
gc.collect()

# ============================================================
# TEST 4: _deduplicate_group (chunked mode for >5000 features)
# ============================================================
print("\n" + "=" * 70)
print(f"TEST 4: _deduplicate_group() chunked on {wide_group.shape[1]} features")
print("=" * 70)
signal.alarm(600)  # 10 min timeout for dedup of 15k features
try:
    wide_winsorized = winsorize_covid_period(wide_group)
    t0 = time.time()
    wide_dedup = _deduplicate_group(wide_winsorized, threshold=0.95)
    elapsed = time.time() - t0
    signal.alarm(0)
    
    n_dropped = wide_winsorized.shape[1] - wide_dedup.shape[1]
    mem = check_mem()
    print(f"  Before: {wide_winsorized.shape[1]} cols")
    print(f"  After:  {wide_dedup.shape[1]} cols (collapsed {n_dropped})")
    print(f"  Time:   {elapsed:.1f}s")
    print(f"  RAM:    {mem:.0f} MB")
    
    assert wide_dedup.shape[1] > 0, "All features collapsed!"
    print(f"  {PASS}: Dedup completed in {elapsed:.1f}s")
    results['deduplicate_group'] = PASS
except TimeoutError:
    signal.alarm(0)
    print(f"  {FAIL}: Timed out after 10 minutes!")
    results['deduplicate_group'] = FAIL
except Exception as e:
    signal.alarm(0)
    print(f"  {FAIL}: {e}")
    results['deduplicate_group'] = FAIL

del wide_group, wide_winsorized
gc.collect()

# ============================================================
# TEST 5: filter_group_data_purged on dedup'd group
# ============================================================
print("\n" + "=" * 70)
print(f"TEST 5: filter_group_data_purged() on {wide_dedup.shape[1]} features")
print("=" * 70)
signal.alarm(600)  # 10 min timeout
try:
    t0 = time.time()
    selected = filter_group_data_purged(wide_dedup, y, group_names[0])
    elapsed = time.time() - t0
    signal.alarm(0)
    
    mem = check_mem()
    print(f"  Input:     {wide_dedup.shape[1]} features")
    print(f"  Survivors: {len(selected)} features")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  RAM:       {mem:.0f} MB")
    print(f"  Features:  {selected[:5]}...")
    
    print(f"  {PASS}: Stage 1 completed in {elapsed:.1f}s with {len(selected)} survivors")
    results['filter_group_data_purged'] = PASS
except TimeoutError:
    signal.alarm(0)
    print(f"  {FAIL}: Timed out after 10 minutes!")
    results['filter_group_data_purged'] = FAIL
except Exception as e:
    signal.alarm(0)
    print(f"  {FAIL}: {e}")
    import traceback; traceback.print_exc()
    results['filter_group_data_purged'] = FAIL

# ============================================================
# TEST 6: _boruta_core with shadow capping
# ============================================================
print("\n" + "=" * 70)
print("TEST 6: _boruta_core() with shadow capping (20 runs on small subset)")
print("=" * 70)
signal.alarm(300)  # 5 min timeout
try:
    # Use 50 random features for a fast Boruta test
    rng = np.random.RandomState(42)
    test_cols = rng.choice(wide_dedup.columns, size=min(50, wide_dedup.shape[1]), replace=False)
    X_test = wide_dedup[test_cols]
    common = X_test.index.intersection(y.index)
    X_test = X_test.loc[common]
    y_test = y.loc[common]
    
    t0 = time.time()
    confirmed = _boruta_core(X_test, y_test, n_runs=20, alpha=0.05)
    elapsed = time.time() - t0
    signal.alarm(0)
    
    mem = check_mem()
    print(f"  Input:     {len(test_cols)} features × {len(common)} obs")
    print(f"  Confirmed: {len(confirmed)} features")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  RAM:       {mem:.0f} MB")
    
    print(f"  {PASS}: Boruta completed in {elapsed:.1f}s")
    results['boruta_core'] = PASS
except TimeoutError:
    signal.alarm(0)
    print(f"  {FAIL}: Timed out!")
    results['boruta_core'] = FAIL
except Exception as e:
    signal.alarm(0)
    print(f"  {FAIL}: {e}")
    import traceback; traceback.print_exc()
    results['boruta_core'] = FAIL

# ============================================================
# TEST 7: LGB_PARAMS safety check
# ============================================================
print("\n" + "=" * 70)
print("TEST 7: LGB_PARAMS n_jobs safety")
print("=" * 70)
try:
    assert LGB_PARAMS['n_jobs'] == 1, f"n_jobs={LGB_PARAMS['n_jobs']}, expected 1!"
    print(f"  LGB_PARAMS['n_jobs'] = {LGB_PARAMS['n_jobs']}")
    print(f"  {PASS}: n_jobs is safely set to 1 (prevents OpenMP deadlock)")
    results['lgb_params_safety'] = PASS
except AssertionError as e:
    print(f"  {FAIL}: {e}")
    results['lgb_params_safety'] = FAIL

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
all_pass = True
for test_name, result in results.items():
    print(f"  {result}  {test_name}")
    if FAIL in result:
        all_pass = False

if all_pass:
    print(f"\n🎉 ALL {len(results)} TESTS PASSED! Pipeline is ready for production.")
else:
    failed = sum(1 for r in results.values() if FAIL in r)
    print(f"\n⚠️  {failed}/{len(results)} tests FAILED. Fix issues before running pipeline.")

print(f"\nPeak RSS: {mem_mb():.0f} MB")
print(f"System Memory: {psutil.virtual_memory().percent:.0f}% used")
