#!/usr/bin/env python3
"""
Predict Next NFP

Production prediction entrypoint that delegates inference to the canonical
Train/train_lightgbm_nfp.py pipeline.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from Train.config import get_model_id
from Train.train_lightgbm_nfp import get_latest_prediction, predict_nfp_mom

logger = setup_logger(__file__, TEMP_DIR)


def _parse_target_month(month_text: str) -> pd.Timestamp:
    """Parse YYYY-MM (or YYYY-MM-DD) into first-of-month Timestamp."""
    month_text = month_text.strip()
    if len(month_text) == 7:
        month_text = f"{month_text}-01"
    return pd.Timestamp(month_text).replace(day=1)


def _format_result(raw_result: Dict, target_source: str) -> Dict:
    """Format canonical prediction output for script JSON/report compatibility."""
    target_month = pd.Timestamp(raw_result["target_month"])
    intervals = raw_result["intervals"]

    return {
        "timestamp": datetime.now().isoformat(),
        "target_month": target_month.strftime("%Y-%m"),
        "target_type": raw_result["target_type"].upper(),
        "release_type": raw_result["release_type"],
        "target_source": target_source,
        "model_id": raw_result["model_id"],
        "prediction": {
            "point_estimate": round(float(raw_result["prediction"]), 0),
            "unit": "thousands of jobs (MoM change)",
            "confidence_intervals": {
                "50%": [round(float(intervals["50%"][0]), 0), round(float(intervals["50%"][1]), 0)],
                "80%": [round(float(intervals["80%"][0]), 0), round(float(intervals["80%"][1]), 0)],
                "95%": [round(float(intervals["95%"][0]), 0), round(float(intervals["95%"][1]), 0)],
            },
        },
        "model_info": {
            "features_used": int(raw_result.get("features_used", 0)),
            "std": float(raw_result.get("std", 0.0)),
            "mean_residual_bias": float(raw_result.get("mean_residual_bias", 0.0)),
        },
    }


def generate_prediction(
    target_type: str = "nsa",
    release_type: str = "first",
    target_source: str = "first_release",
    target_month: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Generate NFP prediction from the canonical train/inference pipeline.
    """
    model_id = get_model_id(target_type, release_type, target_source)
    logger.info(f"Generating prediction using model {model_id.upper()}")

    if target_month:
        month_ts = _parse_target_month(target_month)
        raw_result = predict_nfp_mom(
            month_ts,
            target_type=target_type,
            release_type=release_type,
            target_source=target_source,
        )
    else:
        raw_result = get_latest_prediction(
            target_type=target_type,
            release_type=release_type,
            target_source=target_source,
        )

    result = _format_result(raw_result, target_source=target_source)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved prediction report to {out}")

    ci = result["prediction"]["confidence_intervals"]
    point = result["prediction"]["point_estimate"]
    print(f"\n{'=' * 60}")
    print(f"NFP PREDICTION: {result['target_month']} [{result['model_id'].upper()}]")
    print(f"{'=' * 60}")
    print(f"Point Estimate: {point:+,.0f}K jobs")
    print(f"50% CI: [{ci['50%'][0]:+,.0f}K, {ci['50%'][1]:+,.0f}K]")
    print(f"80% CI: [{ci['80%'][0]:+,.0f}K, {ci['80%'][1]:+,.0f}K]")
    print(f"95% CI: [{ci['95%'][0]:+,.0f}K, {ci['95%'][1]:+,.0f}K]")
    print(f"{'=' * 60}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate NFP prediction")
    parser.add_argument(
        "--target",
        type=str,
        default="nsa",
        choices=["nsa", "sa"],
        help="Target type (default: nsa)",
    )
    parser.add_argument(
        "--release",
        type=str,
        default="first",
        choices=["first", "last"],
        help="Release type (default: first)",
    )
    parser.add_argument(
        "--revised",
        action="store_true",
        help="Use revised target-source model variant",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Predict specific month (YYYY-MM). Defaults to latest available.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for JSON report",
    )

    args = parser.parse_args()
    target_source = "revised" if args.revised else "first_release"

    try:
        generate_prediction(
            target_type=args.target,
            release_type=args.release,
            target_source=target_source,
            target_month=args.month,
            output_path=args.output,
        )
        return 0
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        print(f"\nError: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
