"""
Freeze SA-revised champion/challenger baselines into immutable snapshot folders.

Default mapping for baseline step 1:
- champion_v1  -> _output/sandbox/sa_blend_walkforward
- challenger_v1 -> _output/SA_prediction_revised
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "_output"
DEFAULT_COMPARISON_CSV = (
    DEFAULT_OUTPUT_DIR / "sandbox" / "sa_revised_comparison" / "model_metrics_ranked.csv"
)

DEFAULT_FILES = [
    "backtest_results.csv",
    "summary_statistics.csv",
    "summary_metrics.json",
    "feature_importance.csv",
    "blend_config.json",
    "backtest_predictions.png",
    "summary_table.png",
    "shap_values.png",
    "acf_sa_revised.csv",
    "pacf_sa_revised.csv",
    "acf_error_sa_revised.csv",
    "pacf_error_sa_revised.csv",
    "acf_pacf_diagnostics.png",
    "acf_pacf_sa_revised_and_error.png",
]


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_metrics(src_dir: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    summary_json = src_dir / "summary_metrics.json"
    if summary_json.exists():
        try:
            with summary_json.open("r") as f:
                payload = json.load(f)
            raw = payload.get("overall", payload)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    metrics[k] = _safe_float(v)
        except Exception:
            pass

    summary_csv = src_dir / "summary_statistics.csv"
    if summary_csv.exists():
        try:
            row = pd.read_csv(summary_csv).iloc[0].to_dict()
            for k, v in row.items():
                metrics.setdefault(k, _safe_float(v))
        except Exception:
            pass

    backtest_csv = src_dir / "backtest_results.csv"
    if backtest_csv.exists():
        try:
            bt = pd.read_csv(backtest_csv)
            if {"actual", "predicted"}.issubset(bt.columns):
                valid = bt[bt["actual"].notna() & bt["predicted"].notna()].copy()
                if not valid.empty:
                    actual = valid["actual"].astype(float).values
                    pred = valid["predicted"].astype(float).values
                    metrics.setdefault(
                        "Directional_Accuracy",
                        float(np.mean(np.sign(actual) == np.sign(pred))),
                    )
                    if len(valid) >= 2:
                        accel = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(pred))))
                    else:
                        accel = float("nan")
                    metrics.setdefault("Acceleration_Accuracy", accel)
                    metrics.setdefault("N_Backtest", float(len(valid)))
        except Exception:
            pass

    return metrics


def _extract_comparison_row(src_dir: Path, comparison_csv: Path) -> Dict[str, object]:
    if not comparison_csv.exists():
        return {}
    try:
        df = pd.read_csv(comparison_csv)
    except Exception:
        return {}
    if "folder" not in df.columns:
        return {}

    source_resolved = str(src_dir.resolve())
    match = df[df["folder"].astype(str) == source_resolved]
    if match.empty:
        return {}

    row = match.iloc[0].to_dict()
    out: Dict[str, object] = {}
    for key, value in row.items():
        num = _safe_float(value)
        if np.isfinite(num):
            out[key] = num
        else:
            out[key] = value
    return out


def _copy_snapshot(
    src_dir: Path,
    dst_dir: Path,
    include_files: List[str],
    force: bool,
) -> List[Dict]:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    if dst_dir.exists():
        if force:
            shutil.rmtree(dst_dir)
        else:
            raise FileExistsError(
                f"Destination already exists: {dst_dir}. Re-run with --force to overwrite."
            )

    dst_dir.mkdir(parents=True, exist_ok=True)

    copied: List[Dict] = []
    seen = set()
    for name in include_files:
        src_file = src_dir / name
        if not src_file.exists() or not src_file.is_file():
            continue
        dst_file = dst_dir / name
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        seen.add(name)
        copied.append(
            {
                "name": name,
                "size_bytes": int(dst_file.stat().st_size),
                "sha256": _sha256(dst_file),
            }
        )

    # If the source contains new top-level files not in DEFAULT_FILES, include them too.
    for src_file in sorted(src_dir.iterdir()):
        if not src_file.is_file():
            continue
        if src_file.name in seen:
            continue
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied.append(
            {
                "name": src_file.name,
                "size_bytes": int(dst_file.stat().st_size),
                "sha256": _sha256(dst_file),
            }
        )

    return copied


def _build_entry(
    role_id: str,
    source_dir: Path,
    frozen_dir: Path,
    artifacts: List[Dict],
    comparison_csv: Path,
) -> Dict:
    return {
        "model_id": role_id,
        "source_dir": str(source_dir.resolve()),
        "frozen_dir": str(frozen_dir.resolve()),
        "metrics": _extract_metrics(source_dir),
        "comparison_row": _extract_comparison_row(source_dir, comparison_csv),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze SA revised baseline snapshots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory (default: _output).",
    )
    parser.add_argument(
        "--champion-src",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sandbox" / "sa_blend_walkforward",
        help="Source folder for champion model artifacts.",
    )
    parser.add_argument(
        "--challenger-src",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "SA_prediction_revised",
        help="Source folder for challenger model artifacts.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version label suffix (example: v1, v2).",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=DEFAULT_COMPARISON_CSV,
        help="Comparison ranking CSV used for score metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing frozen baseline version if already present.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    frozen_root = output_dir / "frozen_baselines" / "sa_revised"
    registry_root = output_dir / "model_registry" / "sa_revised"
    frozen_root.mkdir(parents=True, exist_ok=True)
    registry_root.mkdir(parents=True, exist_ok=True)

    champion_id = f"champion_{args.version}"
    challenger_id = f"challenger_{args.version}"
    champion_dst = frozen_root / champion_id
    challenger_dst = frozen_root / challenger_id

    champion_artifacts = _copy_snapshot(
        src_dir=args.champion_src.resolve(),
        dst_dir=champion_dst,
        include_files=DEFAULT_FILES,
        force=args.force,
    )
    challenger_artifacts = _copy_snapshot(
        src_dir=args.challenger_src.resolve(),
        dst_dir=challenger_dst,
        include_files=DEFAULT_FILES,
        force=args.force,
    )

    payload = {
        "registry_id": f"sa_revised_baseline_{args.version}",
        "target": "sa_first_revised",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "Train/sandbox/freeze_sa_revised_baseline.py",
        "argv": sys.argv[1:],
        "champion": _build_entry(
            role_id=champion_id,
            source_dir=args.champion_src.resolve(),
            frozen_dir=champion_dst,
            artifacts=champion_artifacts,
            comparison_csv=args.comparison_csv.resolve(),
        ),
        "challenger": _build_entry(
            role_id=challenger_id,
            source_dir=args.challenger_src.resolve(),
            frozen_dir=challenger_dst,
            artifacts=challenger_artifacts,
            comparison_csv=args.comparison_csv.resolve(),
        ),
    }

    manifest_path = registry_root / f"sa_revised_baseline_{args.version}.json"
    latest_path = registry_root / "sa_revised_baseline_latest.json"
    with manifest_path.open("w") as f:
        json.dump(payload, f, indent=2)
    with latest_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote baseline manifest: {manifest_path}")
    print(f"Updated latest manifest: {latest_path}")
    print(f"Frozen champion artifacts: {champion_dst}")
    print(f"Frozen challenger artifacts: {challenger_dst}")


if __name__ == "__main__":
    main()
