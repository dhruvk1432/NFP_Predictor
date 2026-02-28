"""
Run complete SA revised sandbox suite without touching core pipeline outputs.

Sequence:
1) CatBoost sandbox backtest
2) XGBoost sandbox backtest
3) SA blend sandbox backtest
4) LightGBM variant suite (legacy -> variance-focused)
5) Unified comparison tables + graphs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SANDBOX_DIR = REPO_ROOT / "Train" / "sandbox"


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SA revised sandbox model suite.")
    parser.add_argument("--skip-catboost", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-blend", action="store_true")
    parser.add_argument("--variants", type=str, default="all")
    parser.add_argument("--min-train-rows", type=str, default="120")
    parser.add_argument("--backtest-months", type=str, default="")
    parser.add_argument("--archive-limit", type=str, default="8")
    parser.add_argument("--overlay-top-n", type=str, default="6")
    parser.add_argument("--min-backtest-rows", type=str, default="12")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--freeze-baseline", action="store_true")
    parser.add_argument("--freeze-version", type=str, default="v1")
    parser.add_argument("--freeze-force", action="store_true")

    parser.add_argument("--tune-xgboost", action="store_true")
    parser.add_argument("--xgb-tune-trials", type=str, default="")
    parser.add_argument("--xgb-tune-timeout", type=str, default="")
    parser.add_argument("--xgb-tune-objective", type=str, default="composite")
    parser.add_argument("--xgb-tune-every-steps", type=str, default="")
    parser.add_argument("--xgb-tune-huber", action="store_true")

    parser.add_argument("--tune-blend", action="store_true")
    parser.add_argument("--blend-tune-trials", type=str, default="")
    parser.add_argument("--blend-tune-timeout", type=str, default="")
    parser.add_argument("--blend-tune-objective", type=str, default="composite")
    parser.add_argument("--blend-tune-cv-splits", type=str, default="")

    parser.add_argument("--tune-variants", action="store_true")
    parser.add_argument("--tune-trials", type=str, default="")
    parser.add_argument("--tune-timeout", type=str, default="")
    parser.add_argument("--tune-objective", type=str, default="composite")
    parser.add_argument("--tune-every-steps", type=str, default="")
    parser.add_argument("--no-tune-huber", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_catboost:
        _run([py, str(SANDBOX_DIR / "experiment_catboost_sa_revised.py")])

    if not args.skip_xgboost:
        xgb_cmd = [py, str(SANDBOX_DIR / "experiment_xgboost_sa_revised.py")]
        if args.tune_xgboost:
            xgb_cmd.append("--tune")
        if args.backtest_months.strip():
            xgb_cmd.extend(["--backtest-months", args.backtest_months.strip()])
        if args.min_train_rows.strip():
            xgb_cmd.extend(["--min-train-rows", args.min_train_rows.strip()])
        if args.xgb_tune_trials.strip():
            xgb_cmd.extend(["--tune-trials", args.xgb_tune_trials.strip()])
        if args.xgb_tune_timeout.strip():
            xgb_cmd.extend(["--tune-timeout", args.xgb_tune_timeout.strip()])
        if args.xgb_tune_objective.strip():
            xgb_cmd.extend(["--tune-objective", args.xgb_tune_objective.strip()])
        if args.xgb_tune_every_steps.strip():
            xgb_cmd.extend(["--tune-every-steps", args.xgb_tune_every_steps.strip()])
        if args.xgb_tune_huber:
            xgb_cmd.append("--tune-huber")
        _run(xgb_cmd)

    if not args.skip_blend:
        blend_cmd = [py, str(SANDBOX_DIR / "experiment_sa_blend.py")]
        if args.tune_blend:
            blend_cmd.append("--tune")
        if args.blend_tune_trials.strip():
            blend_cmd.extend(["--tune-trials", args.blend_tune_trials.strip()])
        if args.blend_tune_timeout.strip():
            blend_cmd.extend(["--tune-timeout", args.blend_tune_timeout.strip()])
        if args.blend_tune_objective.strip():
            blend_cmd.extend(["--tune-objective", args.blend_tune_objective.strip()])
        if args.blend_tune_cv_splits.strip():
            blend_cmd.extend(["--tune-cv-splits", args.blend_tune_cv_splits.strip()])
        _run(blend_cmd)

    variant_cmd = [
        py,
        str(SANDBOX_DIR / "experiment_lgbm_sa_revised_variants.py"),
        "--variants",
        args.variants,
        "--min-train-rows",
        args.min_train_rows,
    ]
    if args.backtest_months.strip():
        variant_cmd.extend(["--backtest-months", args.backtest_months.strip()])
    if args.tune_variants:
        variant_cmd.append("--tune")
    if args.tune_trials.strip():
        variant_cmd.extend(["--tune-trials", args.tune_trials.strip()])
    if args.tune_timeout.strip():
        variant_cmd.extend(["--tune-timeout", args.tune_timeout.strip()])
    if args.tune_objective.strip():
        variant_cmd.extend(["--tune-objective", args.tune_objective.strip()])
    if args.tune_every_steps.strip():
        variant_cmd.extend(["--tune-every-steps", args.tune_every_steps.strip()])
    if args.no_tune_huber:
        variant_cmd.append("--no-tune-huber")
    _run(variant_cmd)

    compare_cmd = [
        py,
        str(SANDBOX_DIR / "compare_sa_revised_models.py"),
        "--archive-limit",
        args.archive_limit,
        "--overlay-top-n",
        args.overlay_top_n,
        "--min-backtest-rows",
        args.min_backtest_rows,
    ]
    if not args.no_archive:
        compare_cmd.append("--include-archive")
    _run(compare_cmd)

    if args.freeze_baseline:
        freeze_cmd = [
            py,
            str(SANDBOX_DIR / "freeze_sa_revised_baseline.py"),
            "--version",
            args.freeze_version,
        ]
        if args.freeze_force:
            freeze_cmd.append("--force")
        _run(freeze_cmd)


if __name__ == "__main__":
    main()
