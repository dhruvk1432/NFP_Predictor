"""
Trim master snapshot parquet files to selected-feature overlaps.

For each branch:
    - Load selected features from data/master_snapshots/selected_features_<branch>.json
    - For every snapshot parquet in that branch, keep only columns that are in both:
        (a) that parquet's current columns
        (b) the selected feature set (typically 50)
    - Rewrite parquet atomically
    - Verify post-write columns exactly match the expected overlap

Usage:
    python -m Train.prune_snapshots_to_selected_features
    python -m Train.prune_snapshots_to_selected_features --workers 12
    python -m Train.prune_snapshots_to_selected_features --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import pyarrow.parquet as pq


BRANCH_CONFIG = {
    "nsa_revised": {
        "features_json": "selected_features_nsa_revised.json",
        "snapshots_dir": "nsa/revised/decades",
    },
    "sa_revised": {
        "features_json": "selected_features_sa_revised.json",
        "snapshots_dir": "sa/revised/decades",
    },
}


@dataclass
class FileResult:
    branch: str
    path: Path
    original_count: int
    kept_count: int
    selected_hits: int
    matched_selected: Set[str]
    rewritten: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Keep sanitization behavior aligned with Train.data_loader.sanitize_feature_name.
_SANITIZE_MULTI_CHAR = {
    "+": "plus",
    "%": "pct",
    "&": "_and_",
    "<": "_lt_",
    ">": "_gt_",
}
_SANITIZE_MULTI_RE = re.compile("|".join(re.escape(k) for k in _SANITIZE_MULTI_CHAR))
_SANITIZE_TO_UNDERSCORE = re.compile(r"[|\s\[\]{}\\,()\?/:;!@#$*=.<>]")
_SANITIZE_STRIP_QUOTES = re.compile(r"[\"']")
_SANITIZE_INTERIOR_HYPHEN = re.compile(r"(?<!^)-(?!$)")
_SANITIZE_COLLAPSE = re.compile(r"_+")


def sanitize_feature_name(name: str) -> str:
    """Sanitize feature names exactly like training/reduction code paths."""
    name = _SANITIZE_MULTI_RE.sub(lambda m: _SANITIZE_MULTI_CHAR[m.group()], name)
    name = _SANITIZE_STRIP_QUOTES.sub("", name)
    name = _SANITIZE_INTERIOR_HYPHEN.sub("_", name)
    name = _SANITIZE_TO_UNDERSCORE.sub("_", name)
    name = _SANITIZE_COLLAPSE.sub("_", name).strip("_")
    return name


def _load_selected_features(master_snapshots_dir: Path, json_name: str) -> Set[str]:
    path = master_snapshots_dir / json_name
    payload = json.loads(path.read_text())
    features = payload.get("features")
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise ValueError(f"Invalid feature payload in {path}")
    # Normalize JSON feature names into the same namespace used for snapshot columns.
    sanitized = {sanitize_feature_name(str(x)) for x in features}
    return sanitized


def _collect_branch_files(branch_dir: Path) -> List[Path]:
    if not branch_dir.exists():
        return []
    return sorted(p for p in branch_dir.rglob("*.parquet") if p.is_file())


def _process_file(path: Path, selected_set: Set[str], branch: str, dry_run: bool) -> FileResult:
    original_cols = pq.ParquetFile(path).schema_arrow.names
    sanitized_map = {c: sanitize_feature_name(str(c)) for c in original_cols}
    keep_cols = [c for c in original_cols if sanitized_map[c] in selected_set]
    expected_selected = {sanitized_map[c] for c in keep_cols}

    rewritten = False
    if dry_run:
        return FileResult(
            branch=branch,
            path=path,
            original_count=len(original_cols),
            kept_count=len(keep_cols),
            selected_hits=len(expected_selected),
            matched_selected=expected_selected,
            rewritten=False,
        )

    if keep_cols != original_cols:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        table = pq.read_table(path, columns=keep_cols)
        try:
            pq.write_table(table, tmp_path, compression="snappy")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        rewritten = True

    post_cols = pq.ParquetFile(path).schema_arrow.names
    post_sanitized = {sanitize_feature_name(str(c)) for c in post_cols}
    if post_sanitized != expected_selected:
        raise ValueError(
            f"Verification failed for {path}: expected {len(expected_selected)} selected names, "
            f"got {len(post_sanitized)}"
        )

    return FileResult(
        branch=branch,
        path=path,
        original_count=len(original_cols),
        kept_count=len(keep_cols),
        selected_hits=len(expected_selected),
        matched_selected=expected_selected,
        rewritten=rewritten,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune snapshot columns to selected-feature overlap.")
    parser.add_argument(
        "--master-snapshots-dir",
        type=Path,
        default=_repo_root() / "data" / "master_snapshots",
        help="Path to data/master_snapshots",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 4),
        help="Thread workers for file-level parallelism.",
    )
    parser.add_argument(
        "--branches",
        nargs="*",
        default=list(BRANCH_CONFIG.keys()),
        choices=list(BRANCH_CONFIG.keys()),
        help="Subset of branches to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and verify expected overlaps without rewriting files.",
    )
    args = parser.parse_args()

    master_snapshots_dir = args.master_snapshots_dir.resolve()
    if not master_snapshots_dir.exists():
        raise FileNotFoundError(f"Master snapshots dir not found: {master_snapshots_dir}")

    selected_by_branch: Dict[str, Set[str]] = {}
    files_to_process: List[tuple[str, Path]] = []

    print(f"[INFO] Master snapshots dir: {master_snapshots_dir}")
    for branch in args.branches:
        cfg = BRANCH_CONFIG[branch]
        selected_set = _load_selected_features(master_snapshots_dir, cfg["features_json"])
        selected_by_branch[branch] = selected_set
        branch_dir = master_snapshots_dir / cfg["snapshots_dir"]
        branch_files = _collect_branch_files(branch_dir)
        files_to_process.extend((branch, p) for p in branch_files)
        print(
            f"[INFO] {branch}: selected={len(selected_set)} files={len(branch_files)} dir={branch_dir}"
        )

    total_files = len(files_to_process)
    if total_files == 0:
        print("[INFO] No parquet files found for requested branches.")
        return 0

    branch_stats: Dict[str, Dict[str, int]] = {
        b: {
            "files": 0,
            "rewritten": 0,
            "unchanged": 0,
            "orig_cols_sum": 0,
            "kept_cols_sum": 0,
            "selected_hits_sum": 0,
        }
        for b in args.branches
    }
    selected_seen_by_branch: Dict[str, Set[str]] = {b: set() for b in args.branches}
    errors: List[str] = []

    print(
        f"[INFO] Starting {'dry-run ' if args.dry_run else ''}processing of {total_files} files "
        f"with workers={args.workers}"
    )
    completed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {
            pool.submit(_process_file, path, selected_by_branch[branch], branch, args.dry_run): (branch, path)
            for branch, path in files_to_process
        }
        for future in as_completed(future_map):
            branch, path = future_map[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                errors.append(f"{branch} | {path}: {exc}")
                continue

            stats = branch_stats[result.branch]
            stats["files"] += 1
            stats["rewritten"] += int(result.rewritten)
            stats["unchanged"] += int(not result.rewritten)
            stats["orig_cols_sum"] += result.original_count
            stats["kept_cols_sum"] += result.kept_count
            stats["selected_hits_sum"] += result.selected_hits
            selected_seen_by_branch[result.branch].update(result.matched_selected)

            if completed % 250 == 0 or completed == total_files:
                print(f"[INFO] Progress: {completed}/{total_files}")

    print("\n[INFO] Branch summary")
    for branch in args.branches:
        s = branch_stats[branch]
        if s["files"] == 0:
            print(f"[INFO] {branch}: files=0")
            continue
        avg_orig = s["orig_cols_sum"] / s["files"]
        avg_kept = s["kept_cols_sum"] / s["files"]
        avg_hits = s["selected_hits_sum"] / s["files"]
        selected_total = len(selected_by_branch[branch])
        selected_seen = len(selected_seen_by_branch[branch])
        unmatched = sorted(selected_by_branch[branch] - selected_seen_by_branch[branch])
        print(
            f"[INFO] {branch}: files={s['files']} rewritten={s['rewritten']} unchanged={s['unchanged']} "
            f"avg_orig_cols={avg_orig:.1f} avg_kept_cols={avg_kept:.1f} avg_selected_hits={avg_hits:.1f} "
            f"selected_covered={selected_seen}/{selected_total}"
        )
        if unmatched:
            sample = ", ".join(unmatched[:5])
            print(f"[WARN] {branch}: selected features never matched in any file ({len(unmatched)}). Sample: {sample}")

    if errors:
        print(f"\n[ERROR] {len(errors)} file(s) failed:")
        for msg in errors[:20]:
            print(f"  - {msg}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more")
        return 1

    print("\n[INFO] Completed successfully with full per-file verification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
