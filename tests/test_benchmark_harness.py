import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from utils.benchmark_harness import (
    build_keep_rule_report,
    read_mae_from_summary_csv,
    run_command_timed,
)


def test_read_mae_from_summary_csv(tmp_path):
    path = tmp_path / "summary_statistics.csv"
    pd.DataFrame(
        [
            {"RMSE": 100.0, "MAE": 90.0, "MSE": 10000.0},
            {"RMSE": 80.0, "MAE": 70.0, "MSE": 6400.0},
        ]
    ).to_csv(path, index=False)

    mae = read_mae_from_summary_csv(path)
    assert mae == 70.0


def test_build_keep_rule_report_machine_readable():
    report = build_keep_rule_report(
        baseline_mae=100.0,
        candidate_mae=100.4,
        baseline_runtime_s=100.0,
        candidate_runtime_s=80.0,
    )

    assert report["schema_version"] == "keep_rule_report_v1"
    assert report["decision"]["label"] in {"accept", "reject"}
    assert report["decision"]["keep"] is True
    assert "mae_improvement_pct" in report["deltas_pct"]
    assert "runtime_improvement_pct" in report["deltas_pct"]


def test_run_command_timed_records_runtime():
    cmd = f"{sys.executable} -c \"print('ok')\""
    result = run_command_timed(cmd)
    assert result["returncode"] == 0
    assert result["runtime_s"] >= 0
    assert "ok" in result["stdout"]


def test_cli_writes_json_report(tmp_path):
    script = Path("scripts/benchmark_keep_rule.py").resolve()
    output = tmp_path / "report.json"

    cmd = [
        sys.executable,
        str(script),
        "--baseline-mae", "100",
        "--candidate-mae", "99.4",
        "--baseline-runtime-s", "100",
        "--candidate-runtime-s", "110",
        "--output", str(output),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert output.exists()

    with open(output, "r") as f:
        report = json.load(f)
    assert report["decision"]["keep"] is True
    assert report["decision"]["label"] == "accept"
