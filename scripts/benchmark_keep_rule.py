#!/usr/bin/env python3
"""
Benchmark Keep-Rule Harness

Compares baseline vs candidate MAE/runtime and emits a machine-readable
accept/reject report using should_keep_change.

Modes:
1) Direct metrics:
   python scripts/benchmark_keep_rule.py \
       --baseline-mae 164 --candidate-mae 161 \
       --baseline-runtime-s 120 --candidate-runtime-s 98

2) MAE from summary CSV + timed commands:
   python scripts/benchmark_keep_rule.py \
       --baseline-cmd "python run_baseline.py" \
       --candidate-cmd "python run_candidate.py" \
       --baseline-summary-csv _output/baseline/summary_statistics.csv \
       --candidate-summary-csv _output/candidate/summary_statistics.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from settings import TEMP_DIR, setup_logger
from utils.benchmark_harness import (
    build_keep_rule_report,
    read_mae_from_summary_csv,
    run_command_timed,
    save_report,
)

logger = setup_logger(__file__, TEMP_DIR)


def _resolve_mae(label: str, mae: float | None, summary_csv: str | None) -> float:
    if mae is not None:
        return float(mae)
    if summary_csv:
        return read_mae_from_summary_csv(Path(summary_csv))
    raise ValueError(f"{label}: provide --{label}-mae or --{label}-summary-csv")


def _resolve_runtime(
    label: str,
    runtime_s: float | None,
    cmd: str | None,
    workdir: Path | None,
) -> tuple[float, dict | None]:
    if cmd:
        result = run_command_timed(cmd, workdir=workdir)
        if result["returncode"] != 0:
            raise RuntimeError(
                f"{label} command failed (exit={result['returncode']}): {cmd}\n"
                f"stderr:\n{result['stderr']}"
            )
        return float(result["runtime_s"]), result
    if runtime_s is not None:
        return float(runtime_s), None
    raise ValueError(f"{label}: provide --{label}-runtime-s or --{label}-cmd")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs candidate metrics and apply keep-rule."
    )
    parser.add_argument("--baseline-mae", type=float, default=None)
    parser.add_argument("--candidate-mae", type=float, default=None)
    parser.add_argument("--baseline-summary-csv", type=str, default=None)
    parser.add_argument("--candidate-summary-csv", type=str, default=None)

    parser.add_argument("--baseline-runtime-s", type=float, default=None)
    parser.add_argument("--candidate-runtime-s", type=float, default=None)
    parser.add_argument("--baseline-cmd", type=str, default=None)
    parser.add_argument("--candidate-cmd", type=str, default=None)
    parser.add_argument("--workdir", type=str, default=None)

    parser.add_argument("--min-mae-improvement-pct", type=float, default=0.5)
    parser.add_argument("--min-runtime-improvement-pct", type=float, default=15.0)
    parser.add_argument("--max-mae-loss-for-runtime-pct", type=float, default=0.5)

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: _output/benchmark_reports/keep_rule_report.json",
    )
    parser.add_argument(
        "--fail-on-reject",
        action="store_true",
        help="Exit 2 when decision is reject.",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve() if args.workdir else None

    baseline_runtime_s, baseline_cmd_result = _resolve_runtime(
        "baseline", args.baseline_runtime_s, args.baseline_cmd, workdir
    )
    candidate_runtime_s, candidate_cmd_result = _resolve_runtime(
        "candidate", args.candidate_runtime_s, args.candidate_cmd, workdir
    )

    baseline_mae = _resolve_mae("baseline", args.baseline_mae, args.baseline_summary_csv)
    candidate_mae = _resolve_mae("candidate", args.candidate_mae, args.candidate_summary_csv)

    metadata = {
        "inputs": {
            "baseline_summary_csv": args.baseline_summary_csv,
            "candidate_summary_csv": args.candidate_summary_csv,
        },
        "executions": {
            "baseline": baseline_cmd_result,
            "candidate": candidate_cmd_result,
        },
    }

    report = build_keep_rule_report(
        baseline_mae=baseline_mae,
        candidate_mae=candidate_mae,
        baseline_runtime_s=baseline_runtime_s,
        candidate_runtime_s=candidate_runtime_s,
        min_mae_improvement_pct=args.min_mae_improvement_pct,
        min_runtime_improvement_pct=args.min_runtime_improvement_pct,
        max_mae_loss_for_runtime_pct=args.max_mae_loss_for_runtime_pct,
        metadata=metadata,
    )

    default_output = Path("_output/benchmark_reports/keep_rule_report.json")
    output_path = Path(args.output) if args.output else default_output
    save_report(report, output_path)

    print(json.dumps(report, indent=2))
    logger.info(f"Saved keep-rule benchmark report to {output_path}")

    if args.fail_on_reject and not report["decision"]["keep"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
