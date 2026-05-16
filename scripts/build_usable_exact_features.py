"""Build the canonical exact-feature allow manifest.

The output is intentionally a generation allow-set, not a train-time selected
feature set.  "Useful" means plausibly useful for acceleration, HMM/regime
sidecars, NSA prediction, Kalman/fusion, SA diagnostics or prediction, and other
time-series models.  Explicit rejected/demoted manifests still win over anything
that was selected in a cache or dynamic-selection window.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMNS = ["source", "feature"]

DEFAULT_OUTPUT = REPO_ROOT / "usable_exact_features.csv"
USEFUL_PATH = REPO_ROOT / "useful_exact_features_full.csv"
STILL_AMBIGUOUS_PATH = REPO_ROOT / "ambiguous_still_ambiguous_exact_features.csv"
NON_INTERESTING_PATH = REPO_ROOT / "non_interesting_exact_features.csv"
DEMOTED_PATH = REPO_ROOT / "ambiguous_demoted_exact_features.csv"
SELECTED_FEATURE_ROOTS = (
    REPO_ROOT / "data" / "master_snapshots",
    REPO_ROOT / "data_live_refresh_compare" / "master_snapshots",
)
DYNAMIC_SELECTION_ROOTS = (
    REPO_ROOT / "_output" / "dynamic_selection",
)
DYNAMIC_MIN_WINDOWS = 2
RETENTION_USE_CASES = "|".join(
    [
        "acceleration",
        "hmm",
        "nsa_prediction",
        "kalman",
        "sa_prediction",
        "other_time_series",
    ]
)

LAG_SUFFIXES = (
    ("_lag_12m", "lag_12m"),
    ("_lag_6m", "lag_6m"),
    ("_lag_3m", "lag_3m"),
    ("_lag_1m", "lag_1m"),
    ("_lag12", "lag12"),
    ("_lag6", "lag6"),
    ("_lag1", "lag1"),
)
PRIMARY_SUFFIXES = (
    ("_diff_zscore_12m", "diff_zscore_12m"),
    ("_diff_zscore_3m", "diff_zscore_3m"),
    ("_rolling_mean_3m", "rolling_mean_3m"),
    ("_rolling_std_6m", "rolling_std_6m"),
    ("_zscore_12m", "level_zscore_12m"),
    ("_zscore_3m", "level_zscore_3m"),
    ("_symlog_chg_12m", "symlog_chg_12m"),
    ("_symlog_chg_6m", "symlog_chg_6m"),
    ("_symlog_chg_3m", "symlog_chg_3m"),
    ("_symlog_diff", "symlog_diff"),
    ("_pct_chg", "pct_chg"),
    ("_chg_12m", "chg_12m"),
    ("_chg_6m", "chg_6m"),
    ("_chg_3m", "chg_3m"),
    ("_diff", "diff"),
    ("_symlog", "symlog"),
)


def _read_csv(path: Path, *, required: bool) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required manifest: {path}")
        return pd.DataFrame(columns=KEY_COLUMNS)
    frame = pd.read_csv(path, low_memory=False)
    missing = [col for col in KEY_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing key columns: {missing}")
    return frame


def _feature_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if frame.empty:
        return set()
    clean = frame.dropna(subset=KEY_COLUMNS)
    return set(zip(clean["source"].astype(str), clean["feature"].astype(str)))


def _strip_suffix(value: str, suffixes: tuple[tuple[str, str], ...]) -> tuple[str, str | None]:
    for suffix, label in suffixes:
        if value.endswith(suffix):
            return value[: -len(suffix)], label
    return value, None


def _parse_transform_metadata(feature: str) -> dict[str, str]:
    base, lag = _strip_suffix(str(feature), LAG_SUFFIXES)
    base, transform = _strip_suffix(base, PRIMARY_SUFFIXES)
    return {
        "base_series": base,
        "transform_family": transform or "raw",
        "lag": lag or "none",
    }


def _with_decision_columns(
    frame: pd.DataFrame,
    *,
    bucket: str,
    decision: str,
    source_file: Path,
) -> pd.DataFrame:
    out = frame.copy()
    out["usable_bucket"] = bucket
    out["usable_decision"] = decision
    out["usable_source_file"] = source_file.name
    out["retention_use_cases"] = RETENTION_USE_CASES
    return out


def _classify_source(feature: str) -> str:
    feature = str(feature)
    if feature.startswith("total_"):
        return "FRED_Employment_NSA" if "_nsa" in feature else "FRED_Employment_SA"
    if feature.startswith(("Treasury_", "FedFunds_", "SOFR_", "WTI_Crude_", "NatGas_", "Gold_", "Copper_", "DollarIndex_", "EuroFX_", "YenFX_", "SP500_Futures_")):
        return "Futures"
    if feature.startswith(("NFP_Forecast_", "Economist_", "economist_")):
        return "EconomistPanel"
    if feature.startswith(("sanagap_",)):
        return "SA_NSA_Gap"
    if feature.startswith(("is_", "month_", "quarter_", "year", "weeks_since_", "nfp_", "rev_master_")):
        return "DerivedControls"
    if feature.startswith(("NOAA_", "noaa_", "storm_", "hurricane_")):
        return "NOAA"
    if feature.startswith(("Consumer_Mood", "Prosper_", "Consumer_Spending")):
        return "Prosper"
    if feature.startswith(("ADP_", "adp_")):
        return "ADP"
    if feature.startswith(("AHE_", "AWH_", "CB_", "Challenger_", "Empire_", "Housing_", "ISM_", "Industrial_", "NFP_Consensus", "Retail_", "UMich_")):
        return "Unifier"
    if feature.startswith(("CCNSA_", "CCSA_", "Credit_", "Financial_", "Oil_", "SP500_", "VIX_", "Weekly_", "Yield_", "WEI_", "regime_")):
        return "FRED_Exogenous"
    return "Unknown"


def _selected_feature_usage() -> dict[tuple[str, str], dict[str, set[str]]]:
    usage: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"branches": set(), "cache_files": set()}
    )
    for root in SELECTED_FEATURE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/selected_features_*.json")):
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            branch = f"{payload.get('target_cat') or 'unknown'}_{payload.get('target_source') or 'unknown'}"
            for prefixed_feature in payload.get("features", []):
                if "::" not in str(prefixed_feature):
                    continue
                source, feature = str(prefixed_feature).split("::", 1)
                entry = usage[(source, feature)]
                entry["branches"].add(branch)
                entry["cache_files"].add(path.relative_to(REPO_ROOT).as_posix())
    return usage


def _dynamic_selection_usage() -> dict[tuple[str, str], dict[str, set[str]]]:
    usage: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"branches": set(), "windows": set(), "selection_files": set()}
    )
    for root in DYNAMIC_SELECTION_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/*.json")):
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            branch = path.parent.name
            window = str(payload.get("step_date") or path.stem)
            features = payload.get("selected_features") or payload.get("features") or []
            for feature in features:
                feature = str(feature)
                source = _classify_source(feature)
                if source == "Unknown":
                    continue
                entry = usage[(source, feature)]
                entry["branches"].add(branch)
                entry["windows"].add(window)
                entry["selection_files"].add(path.relative_to(REPO_ROOT).as_posix())
    return usage


def build_usable_exact_features() -> pd.DataFrame:
    useful = _with_decision_columns(
        _read_csv(USEFUL_PATH, required=True),
        bucket="useful",
        decision="allow",
        source_file=USEFUL_PATH,
    )
    still_ambiguous = _with_decision_columns(
        _read_csv(STILL_AMBIGUOUS_PATH, required=True),
        bucket="still_ambiguous",
        decision="review_allow",
        source_file=STILL_AMBIGUOUS_PATH,
    )

    blocked = pd.concat(
        [
            _read_csv(NON_INTERESTING_PATH, required=True),
            _read_csv(DEMOTED_PATH, required=False),
        ],
        ignore_index=True,
    )
    blocked_keys = _feature_keys(blocked)

    selected_usage = _selected_feature_usage()
    dynamic_usage = _dynamic_selection_usage()
    existing_keys = _feature_keys(pd.concat([useful, still_ambiguous], ignore_index=True))
    selected_rows = []
    for source, feature in sorted(selected_usage):
        key = (source, feature)
        if key in existing_keys or key in blocked_keys:
            continue
        selected_rows.append(
            {
                "source": source,
                "feature": feature,
                "category": "selected_outside_audit",
                "usable_bucket": "selected_outside_audit",
                "usable_decision": "protect_selected_not_rejected",
                "usable_source_file": "selected_features_*.json",
                "retention_use_cases": RETENTION_USE_CASES,
                **_parse_transform_metadata(feature),
            }
        )
    dynamic_rows = []
    for source, feature in sorted(dynamic_usage):
        key = (source, feature)
        if key in existing_keys or key in blocked_keys:
            continue
        usage = dynamic_usage[key]
        if len(usage["windows"]) < DYNAMIC_MIN_WINDOWS:
            continue
        dynamic_rows.append(
            {
                "source": source,
                "feature": feature,
                "category": "recurring_dynamic_selection",
                "usable_bucket": "dynamic_selection_evidence",
                "usable_decision": "protect_recurring_dynamic_not_rejected",
                "usable_source_file": "dynamic_selection/*.json",
                "retention_use_cases": RETENTION_USE_CASES,
                "dynamic_windows": len(usage["windows"]),
                "dynamic_branches": "|".join(sorted(usage["branches"])),
                "dynamic_evidence_files": "|".join(sorted(usage["selection_files"])[:20]),
                **_parse_transform_metadata(feature),
            }
        )

    usable = pd.concat(
        [useful, still_ambiguous, pd.DataFrame(selected_rows), pd.DataFrame(dynamic_rows)],
        ignore_index=True,
        sort=False,
    )
    overlap = _feature_keys(usable) & blocked_keys
    if overlap:
        sample = "\n".join(f"{source},{feature}" for source, feature in sorted(overlap)[:20])
        raise ValueError(f"Usable manifest overlaps rejected features:\n{sample}")

    dupes = usable.duplicated(subset=KEY_COLUMNS, keep=False)
    if bool(dupes.any()):
        sample = usable.loc[dupes, KEY_COLUMNS].drop_duplicates().head(20)
        raise ValueError("Duplicate usable source/feature keys:\n" + sample.to_string(index=False))

    for col in ["base_series", "transform_family", "lag"]:
        if col not in usable.columns:
            usable[col] = ""
    missing_meta = usable["base_series"].isna() | (usable["base_series"].astype(str) == "")
    if bool(missing_meta.any()):
        parsed = usable.loc[missing_meta, "feature"].astype(str).map(_parse_transform_metadata)
        for col in ["base_series", "transform_family", "lag"]:
            usable.loc[missing_meta, col] = [item[col] for item in parsed]

    return usable.sort_values(["source", "base_series", "transform_family", "lag", "feature"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    usable = build_usable_exact_features()
    output.parent.mkdir(parents=True, exist_ok=True)
    usable.to_csv(output, index=False)
    print(f"Wrote {len(usable):,} usable exact features to {output}")
    print(usable["usable_bucket"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
