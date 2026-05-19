"""PIT-safe HMM labor-regime sidecar.

The HMM is not a production point forecast.  It emits latent-regime, transition
risk, confidence, and acceleration-prior signals that can be replayed as gated
features, Kalman observations, or router/noise inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.sidecars.common import (
    feature_audit_from_frame,
    sidecar_branch_root,
    write_sidecar_artifacts,
)
from experiments.sidecars.feature_matrix import (
    build_sidecar_design,
    select_numeric_feature_cols,
    source_map_for_columns,
)

try:
    from settings import DATA_PATH, OUTPUT_DIR
except RuntimeError:
    DATA_PATH = Path("data")
    OUTPUT_DIR = Path("_output")


DEFAULT_TARGET_TYPE = "sa"
DEFAULT_MODEL_ID = "hmm_labor_regime"
LEGACY_DEFAULT_OUTPUT_DIR = (
    OUTPUT_DIR / "sidecars" / "local_sidecar_once" / "hmm_labor_regime"
)


def _default_target_path(target_type: str) -> Path:
    if target_type == "nsa":
        return DATA_PATH / "NFP_target" / "y_nsa_revised.parquet"
    return DATA_PATH / "NFP_target" / "y_sa_revised.parquet"


def _default_target_space(target_type: str) -> str:
    return "nsa_revised" if target_type == "nsa" else "sa_revised"


def _resolve_output_dir(explicit: Path | None, target_type: str, run_id: str) -> Path:
    if explicit is not None:
        return explicit
    target_type = str(target_type).strip().lower()
    if target_type == "sa":
        return sidecar_branch_root(OUTPUT_DIR, "sa") / run_id / "hmm_labor_regime"
    return LEGACY_DEFAULT_OUTPUT_DIR


DEFAULT_TARGET_PATH = _default_target_path(DEFAULT_TARGET_TYPE)
DEFAULT_OUTPUT_DIR = LEGACY_DEFAULT_OUTPUT_DIR


def _state_mean_acceleration(train: pd.DataFrame, states: np.ndarray) -> dict[int, float]:
    tmp = train[["actual_accel"]].copy()
    tmp["state"] = states
    means = tmp.groupby("state")["actual_accel"].mean()
    return {int(k): float(v) for k, v in means.items()}


def _entropy(prob: np.ndarray) -> float:
    p = np.asarray(prob, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(max(len(prob), 2)))


def _select_step_features(
    train: pd.DataFrame,
    row: pd.Series,
    feature_cols: list[str],
    *,
    min_non_nan: int,
    max_features: int,
) -> list[str]:
    usable: list[tuple[int, float, str]] = []
    for col in feature_cols:
        s = pd.to_numeric(train[col], errors="coerce")
        count = int(s.notna().sum())
        if count < int(min_non_nan):
            continue
        var = float(s.var(skipna=True))
        if not np.isfinite(var) or var <= 1e-12:
            continue
        if pd.isna(row.get(col, np.nan)) and s.notna().sum() == 0:
            continue
        usable.append((count, var, col))
    usable.sort(reverse=True)
    return [col for _, _, col in usable[: int(max_features)]]


def run_hmm_acceleration_sidecar(
    *,
    target_path: Path | None = None,
    output_dir: Path | None = None,
    start: str = "2000-01",
    min_train: int = 72,
    n_components: int = 2,
    covariance_type: str = "full",
    target_space: str | None = None,
    include_snapshots: bool = True,
    max_snapshot_columns: int = 160,
    max_features: int = 30,
    model_id: str = DEFAULT_MODEL_ID,
    target_type: str = DEFAULT_TARGET_TYPE,
    run_id: str = "local_sidecar_once",
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    target_type = str(target_type).strip().lower()
    if target_type not in {"nsa", "sa"}:
        raise ValueError(f"target_type must be 'nsa' or 'sa'; got {target_type!r}")
    target_path = target_path or _default_target_path(target_type)
    target_space = target_space or _default_target_space(target_type)
    output_dir = _resolve_output_dir(output_dir, target_type, run_id)
    design = build_sidecar_design(
        target_space=target_space,
        target_path=target_path,
        include_snapshots=include_snapshots,
        max_snapshot_columns=max_snapshot_columns,
    )
    design = design.dropna(subset=["prev_mom", "actual_mom", "actual_accel"]).reset_index(drop=True)
    feature_cols = select_numeric_feature_cols(design)
    start_ts = pd.Timestamp(start).to_period("M").to_timestamp()

    rows: list[dict[str, object]] = []
    for idx, row in design.iterrows():
        ds = pd.Timestamp(row["ds"])
        if ds < start_ts:
            continue
        train = design.iloc[:idx].dropna(subset=["actual_accel", "prev_mom", "actual_mom"]).copy()
        if len(train) < int(min_train):
            continue
        step_features = _select_step_features(
            train,
            row,
            feature_cols,
            min_non_nan=max(12, min(int(min_train), 48)),
            max_features=max_features,
        )
        if len(step_features) < 3:
            continue

        med = train[step_features].median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_train_raw = train[step_features].replace([np.inf, -np.inf], np.nan).fillna(med)
        X_pred_raw = pd.DataFrame([row[step_features].to_dict()])[step_features]
        X_pred_raw = X_pred_raw.replace([np.inf, -np.inf], np.nan).fillna(med)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw.to_numpy(dtype=float))
        X_pred = scaler.transform(X_pred_raw.to_numpy(dtype=float))

        model = GaussianHMM(
            n_components=int(n_components),
            covariance_type=covariance_type,
            n_iter=200,
            random_state=42,
            min_covar=1e-3,
        )
        model.fit(X_train)
        train_states = model.predict(X_train)
        state_means = _state_mean_acceleration(train, train_states)
        state_prob = model.predict_proba(X_pred)[0]
        state = int(np.argmax(state_prob))
        predicted_accel = float(
            sum(float(state_prob[s]) * state_means.get(int(s), 0.0) for s in range(len(state_prob)))
        )
        predicted_mom = float(row["prev_mom"] + predicted_accel)
        proba_up = float(
            sum(float(state_prob[s]) for s in range(len(state_prob)) if state_means.get(int(s), 0.0) > 0)
        )
        p_stay = float(model.transmat_[state, state]) if hasattr(model, "transmat_") else np.nan
        transition_risk = 1.0 - p_stay if np.isfinite(p_stay) else np.nan
        expected_duration = 1.0 / max(1.0 - p_stay, 1e-6) if np.isfinite(p_stay) else np.nan
        rows.append(
            {
                "ds": ds,
                "trained_through": pd.Timestamp(train["ds"].max()),
                "regime_state": state,
                "regime_entropy": _entropy(state_prob),
                "regime_transition_risk": transition_risk,
                "regime_expected_duration": expected_duration,
                "regime_state_mean_accel": state_means.get(state, np.nan),
                "predicted_accel": predicted_accel,
                "predicted_accel_sign": float(np.sign(predicted_accel)),
                "predicted_accel_proba_up": proba_up,
                "confidence": float(abs(proba_up - 0.5) * 2.0),
                "uncertainty": float(1.0 - abs(proba_up - 0.5) * 2.0),
                "actual_accel": float(row["actual_accel"]),
                "predicted_mom": predicted_mom,
                "actual_mom": float(row["actual_mom"]),
                "prev_mom": float(row["prev_mom"]),
                "n_features": int(len(step_features)),
            }
        )
        for s, p in enumerate(state_prob):
            rows[-1][f"regime_prob_{s}"] = float(p)

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("HMM sidecar produced no predictions; lower --min-train or check target data")
    audit = feature_audit_from_frame(
        design,
        feature_cols,
        source_map=source_map_for_columns(feature_cols),
    )
    _, metrics = write_sidecar_artifacts(
        output_dir=output_dir,
        model_id=model_id,
        target_space=target_space,
        predictions=results,
        feature_audit=audit,
        config={
            "start": start,
            "min_train": int(min_train),
            "n_components": int(n_components),
            "covariance_type": covariance_type,
            "include_snapshots": bool(include_snapshots),
            "max_snapshot_columns": int(max_snapshot_columns),
            "max_features": int(max_features),
        },
        extra_metrics={
            "min_train": int(min_train),
            "n_components": int(n_components),
            "covariance_type": covariance_type,
            "mean_abs_predicted_accel": float(np.mean(np.abs(results["predicted_accel"]))),
            "mean_abs_actual_accel": float(np.mean(np.abs(results["actual_accel"]))),
        },
        data_paths={"target_path": str(target_path)},
        write_legacy={
            "hmm_acceleration_predictions.csv": "predictions",
            "hmm_acceleration_metrics.json": "metrics",
        },
    )
    return results, metrics


def run_hmm_grid(
    *,
    output_dir: Path,
    target_path: Path | None = None,
    target_space: str | None = None,
    start: str = "2000-01",
    min_train: int = 72,
    include_snapshots: bool = True,
    target_type: str = DEFAULT_TARGET_TYPE,
    run_id: str = "local_sidecar_once",
) -> pd.DataFrame:
    rows = []
    for n_components in (2, 3, 4):
        for covariance_type in ("diag", "full"):
            model_id = f"hmm_k{n_components}_{covariance_type}"
            try:
                _, metrics = run_hmm_acceleration_sidecar(
                    target_path=target_path,
                    output_dir=output_dir / model_id,
                    start=start,
                    min_train=min_train,
                    n_components=n_components,
                    covariance_type=covariance_type,
                    target_space=target_space,
                    include_snapshots=include_snapshots,
                    model_id=model_id,
                    target_type=target_type,
                    run_id=run_id,
                )
                rows.append({"model_id": model_id, **metrics})
            except Exception as exc:
                rows.append({"model_id": model_id, "error": str(exc)})
    report = pd.DataFrame(rows)
    sort_cols = [c for c in ("acceleration_accuracy", "mae") if c in report.columns]
    if sort_cols:
        report = report.sort_values(sort_cols, ascending=[False, True][: len(sort_cols)])
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "model_selection_report.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-path", type=Path, default=None,
                        help="Explicit target parquet; defaults from --target-type.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Explicit output dir; SA falls back to "
                             "sidecars/sa/<run_id>/hmm_labor_regime.")
    parser.add_argument("--target-type", default=DEFAULT_TARGET_TYPE,
                        choices=["nsa", "sa"],
                        help="Branch subtree the artifact lands under.")
    parser.add_argument("--run-id", default="local_sidecar_once",
                        help="Sidecar run-id directory under the branch subtree.")
    parser.add_argument("--start", default="2000-01")
    parser.add_argument("--min-train", type=int, default=72)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--covariance-type", choices=["diag", "full"], default="full")
    parser.add_argument("--target-space", default=None, choices=["sa_revised", "nsa_revised"],
                        help="Target space override; defaults from --target-type.")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--max-snapshot-columns", type=int, default=160)
    parser.add_argument("--max-features", type=int, default=30)
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    if args.grid:
        report = run_hmm_grid(
            output_dir=args.output_dir if args.output_dir is not None
                       else _resolve_output_dir(None, args.target_type, args.run_id),
            target_path=args.target_path,
            target_space=args.target_space,
            start=args.start,
            min_train=args.min_train,
            include_snapshots=not args.no_snapshots,
            target_type=args.target_type,
            run_id=args.run_id,
        )
        print(report.to_string(index=False))
    else:
        _, metrics = run_hmm_acceleration_sidecar(
            target_path=args.target_path,
            output_dir=args.output_dir,
            start=args.start,
            min_train=args.min_train,
            n_components=args.n_components,
            covariance_type=args.covariance_type,
            target_space=args.target_space,
            include_snapshots=not args.no_snapshots,
            max_snapshot_columns=args.max_snapshot_columns,
            max_features=args.max_features,
            target_type=args.target_type,
            run_id=args.run_id,
        )
        import json

        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
