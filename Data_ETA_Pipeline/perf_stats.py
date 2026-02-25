"""
Lightweight performance instrumentation utilities for ETL/prepare stages.

Usage:
    - Enable with: NFP_PERF=1
    - Keep disabled (default) for zero behavior change and near-zero overhead.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import resource
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

try:
    import psutil
except Exception:  # pragma: no cover - defensive import guard
    psutil = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - defensive import guard
    pd = None

try:
    import requests
except Exception:  # pragma: no cover - defensive import guard
    requests = None


_ENABLED = os.getenv("NFP_PERF", "").strip() == "1"
_LOCK = threading.Lock()
_TIMERS: list[dict] = []
_COUNTERS: Counter = Counter()
_HOOKS_INSTALLED = False
_ATEXIT_REGISTERED: set[str] = set()


def is_perf_enabled() -> bool:
    return _ENABLED


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rss_bytes() -> int:
    if psutil is None:
        return 0
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


def _peak_rss_bytes() -> int:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return 0
    # macOS reports bytes; Linux reports KB
    if sys.platform == "darwin":
        return int(ru)
    return int(ru * 1024)


def _safe_json_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _record_phase(
    name: str,
    wall_s: float,
    cpu_s: float,
    rss_delta_mb: float,
    peak_rss_delta_mb: float,
    tags: dict,
) -> None:
    if not _ENABLED:
        return
    record = {
        "name": name,
        "wall_s": float(wall_s),
        "cpu_s": float(cpu_s),
        "rss_delta_mb": float(rss_delta_mb),
        "peak_rss_delta_mb": float(peak_rss_delta_mb),
        "pid": os.getpid(),
        "timestamp_utc": _utc_now_iso(),
        "tags": {k: _safe_json_value(v) for k, v in tags.items()},
    }
    with _LOCK:
        _TIMERS.append(record)


@contextmanager
def perf_phase(name: str, **tags):
    if not _ENABLED:
        yield
        return

    t0 = time.perf_counter()
    c0 = time.process_time()
    r0 = _rss_bytes()
    p0 = _peak_rss_bytes()
    try:
        yield
    finally:
        _record_phase(
            name=name,
            wall_s=time.perf_counter() - t0,
            cpu_s=time.process_time() - c0,
            rss_delta_mb=(_rss_bytes() - r0) / (1024.0 * 1024.0),
            peak_rss_delta_mb=(_peak_rss_bytes() - p0) / (1024.0 * 1024.0),
            tags=tags,
        )


def profiled(name: str | None = None):
    def _decorator(func):
        phase_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            with perf_phase(phase_name):
                return func(*args, **kwargs)

        return _wrapped

    return _decorator


def inc_counter(metric: str, amount: int = 1) -> None:
    if not _ENABLED:
        return
    with _LOCK:
        _COUNTERS[metric] += int(amount)


def _add_bytes(metric: str, amount: int) -> None:
    if not _ENABLED:
        return
    if amount <= 0:
        return
    with _LOCK:
        _COUNTERS[metric] += int(amount)


def _path_size(path_like) -> int:
    try:
        p = Path(path_like)
        if p.exists() and p.is_file():
            return int(p.stat().st_size)
    except Exception:
        return 0
    return 0


def install_hooks() -> None:
    """
    Install optional hooks for parquet I/O and HTTP call counting.
    No-op when NFP_PERF is disabled.
    """
    global _HOOKS_INSTALLED
    if not _ENABLED or _HOOKS_INSTALLED:
        return

    if pd is not None:
        original_read_parquet = pd.read_parquet
        original_to_parquet = pd.DataFrame.to_parquet

        def _read_parquet_hook(path, *args, **kwargs):
            size = _path_size(path)
            if size > 0:
                inc_counter("parquet_files_read", 1)
                _add_bytes("parquet_read_bytes", size)
            return original_read_parquet(path, *args, **kwargs)

        def _to_parquet_hook(self, path, *args, **kwargs):
            out = original_to_parquet(self, path, *args, **kwargs)
            size = _path_size(path)
            if size > 0:
                inc_counter("parquet_files_written", 1)
                _add_bytes("parquet_written_bytes", size)
            return out

        pd.read_parquet = _read_parquet_hook
        pd.DataFrame.to_parquet = _to_parquet_hook

    if requests is not None:
        original_request = requests.sessions.Session.request

        @functools.wraps(original_request)
        def _request_hook(self, method, url, *args, **kwargs):
            inc_counter("api_calls_total", 1)
            host = urlparse(str(url)).netloc or "unknown"
            inc_counter(f"api_calls_host::{host}", 1)
            return original_request(self, method, url, *args, **kwargs)

        requests.sessions.Session.request = _request_hook

    _HOOKS_INSTALLED = True


def reset_perf_data() -> None:
    if not _ENABLED:
        return
    with _LOCK:
        _TIMERS.clear()
        _COUNTERS.clear()


def _summarize_timers() -> list[dict]:
    by_name: dict[str, dict] = {}
    for row in _TIMERS:
        name = row["name"]
        cur = by_name.setdefault(
            name,
            {
                "name": name,
                "calls": 0,
                "wall_s_total": 0.0,
                "cpu_s_total": 0.0,
                "wall_s_max": 0.0,
                "cpu_s_max": 0.0,
            },
        )
        cur["calls"] += 1
        cur["wall_s_total"] += row["wall_s"]
        cur["cpu_s_total"] += row["cpu_s"]
        cur["wall_s_max"] = max(cur["wall_s_max"], row["wall_s"])
        cur["cpu_s_max"] = max(cur["cpu_s_max"], row["cpu_s"])
    return sorted(by_name.values(), key=lambda x: x["wall_s_total"], reverse=True)


def dump_perf_json(
    stage_name: str,
    output_dir: str | Path = "_temp/perf",
    extra: dict | None = None,
    reset: bool = False,
) -> Path | None:
    if not _ENABLED:
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stage_name}_pid{os.getpid()}_{ts}.json"

    with _LOCK:
        payload = {
            "stage_name": stage_name,
            "pid": os.getpid(),
            "generated_at_utc": _utc_now_iso(),
            "counters": dict(_COUNTERS),
            "timers": sorted(_TIMERS, key=lambda x: x["wall_s"], reverse=True),
            "summary_by_name": _summarize_timers(),
            "extra": extra or {},
        }

    out_path.write_text(json.dumps(payload, indent=2))

    if reset:
        reset_perf_data()

    return out_path


def register_atexit_dump(stage_name: str, output_dir: str | Path = "_temp/perf") -> None:
    if not _ENABLED:
        return
    key = f"{os.getpid()}::{stage_name}"
    if key in _ATEXIT_REGISTERED:
        return
    _ATEXIT_REGISTERED.add(key)

    def _dump():
        dump_perf_json(stage_name=stage_name, output_dir=output_dir, reset=False)

    atexit.register(_dump)

