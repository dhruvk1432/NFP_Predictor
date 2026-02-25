#!/usr/bin/env python3
"""
Summarize NFP perf JSON dumps into actionable stage/hotspot tables.

Usage:
    python Data_ETA_Pipeline/perf_summary.py _temp/perf/*.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def _load_timers_and_counters(paths: list[Path]) -> tuple[list[dict], dict[str, int]]:
    timers: list[dict] = []
    counters: dict[str, int] = defaultdict(int)

    for path in paths:
        payload = json.loads(path.read_text())
        timers.extend(payload.get("timers", []))
        for k, v in payload.get("counters", {}).items():
            counters[k] += int(v)
    return timers, dict(counters)


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    by_name: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        name = str(row.get("name", "unknown"))
        by_name[name].append(float(row.get("wall_s", 0.0)))

    out = []
    for name, vals in by_name.items():
        vals_sorted = sorted(vals)
        total = sum(vals_sorted)
        calls = len(vals_sorted)
        out.append(
            {
                "name": name,
                "calls": calls,
                "wall_s_total": total,
                "wall_s_avg": total / calls if calls else 0.0,
                "wall_s_median": median(vals_sorted) if vals_sorted else 0.0,
                "wall_s_max": vals_sorted[-1] if vals_sorted else 0.0,
            }
        )
    return sorted(out, key=lambda x: x["wall_s_total"], reverse=True)


def _group_key(name: str, depth: int) -> str:
    parts = name.split(".")
    return ".".join(parts[:depth]) if len(parts) >= depth else name


def _aggregate_groups(rows: list[dict], depth: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[_group_key(str(row.get("name", "unknown")), depth)].append(row)

    out = []
    for grp, grp_rows in grouped.items():
        total = sum(float(r.get("wall_s", 0.0)) for r in grp_rows)
        calls = len(grp_rows)
        out.append(
            {
                "name": grp,
                "calls": calls,
                "wall_s_total": total,
                "wall_s_avg": total / calls if calls else 0.0,
                "wall_s_median": median(float(r.get("wall_s", 0.0)) for r in grp_rows),
                "wall_s_max": max(float(r.get("wall_s", 0.0)) for r in grp_rows),
            }
        )
    return sorted(out, key=lambda x: x["wall_s_total"], reverse=True)


def _leaf_names(names: list[str]) -> set[str]:
    leaves = set(names)
    for name in names:
        prefix = f"{name}."
        if any(other.startswith(prefix) for other in names if other != name):
            leaves.discard(name)
    return leaves


def _print_table(title: str, rows: list[dict], limit: int) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'total_s':>10} {'calls':>7} {'avg_s':>10} {'median_s':>10} {'max_s':>10}  name")
    for row in rows[:limit]:
        print(
            f"{row['wall_s_total']:10.3f} {row['calls']:7d} "
            f"{row['wall_s_avg']:10.3f} {row['wall_s_median']:10.3f} "
            f"{row['wall_s_max']:10.3f}  {row['name']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NFP perf JSON outputs")
    parser.add_argument("paths", nargs="+", help="Perf JSON file(s)")
    parser.add_argument("--top", type=int, default=25, help="Rows to show per table")
    parser.add_argument("--group-depth", type=int, default=3, help="Prefix depth for grouped summary")
    args = parser.parse_args()

    paths = []
    for raw in args.paths:
        if any(ch in raw for ch in ["*", "?", "["]):
            paths.extend(sorted(Path(".").glob(raw)))
        else:
            paths.append(Path(raw))
    paths = [p for p in paths if p.exists()]

    if not paths:
        raise SystemExit("No perf JSON files found.")

    timers, counters = _load_timers_and_counters(paths)
    if not timers:
        raise SystemExit("No timers found in provided perf JSON files.")

    by_name = _aggregate_rows(timers)
    by_group = _aggregate_groups(timers, depth=max(1, args.group_depth))
    leafs = _leaf_names([r["name"] for r in by_name])
    leaf_rows = [r for r in by_name if r["name"] in leafs]
    leaf_rows = sorted(leaf_rows, key=lambda x: x["wall_s_total"], reverse=True)

    print(f"Loaded {len(paths)} file(s), {len(timers)} timer rows")
    _print_table("Grouped By Prefix", by_group, args.top)
    _print_table("By Exact Timer Name", by_name, args.top)
    _print_table("Leaf-ish Hotspots", leaf_rows, args.top)

    if counters:
        print("\nTop Counters")
        print("------------")
        for key, value in sorted(counters.items(), key=lambda kv: kv[1], reverse=True)[: max(args.top, 10)]:
            print(f"{value:>12d}  {key}")


if __name__ == "__main__":
    main()
