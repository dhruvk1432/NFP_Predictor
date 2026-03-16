"""
One-time script to trim pre-1990 rows from existing master snapshot parquets.

Rewrites each parquet in-place, keeping only rows where date >= 1990-01-01.
"""
import pandas as pd
from pathlib import Path
import sys

DATA_START_FLOOR = pd.Timestamp("1990-01-01")
MASTER_BASE = Path(__file__).resolve().parent.parent / "data" / "master_snapshots"

PATHS = [
    MASTER_BASE / "_unified" / "decades",
    MASTER_BASE / "nsa" / "revised" / "decades",
    MASTER_BASE / "sa" / "revised" / "decades",
]


def main():
    total_files = 0
    total_trimmed = 0
    total_rows_removed = 0

    for base_path in PATHS:
        if not base_path.exists():
            print(f"SKIP (not found): {base_path}")
            continue

        parquets = sorted(base_path.rglob("*.parquet"))
        print(f"\n{base_path.relative_to(MASTER_BASE)}: {len(parquets)} files")

        for pq in parquets:
            total_files += 1
            try:
                df = pd.read_parquet(pq)
            except Exception as e:
                print(f"  ERROR reading {pq.name}: {e}")
                continue

            if "date" not in df.columns:
                continue

            before = len(df)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= DATA_START_FLOOR]
            after = len(df)

            removed = before - after
            if removed > 0:
                df.to_parquet(pq, index=False)
                total_trimmed += 1
                total_rows_removed += removed
                print(f"  {pq.name}: {before} -> {after} rows (-{removed})")

    print(f"\nDone. Files processed: {total_files}, trimmed: {total_trimmed}, "
          f"total rows removed: {total_rows_removed}")


if __name__ == "__main__":
    main()
