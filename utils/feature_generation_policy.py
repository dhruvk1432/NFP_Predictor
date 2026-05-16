"""Feature-generation culling policy.

This module is intentionally independent from model training.  It decides which
generated feature names should be written into source/master snapshots; dynamic
feature selection then works over the smaller emitted universe.

Retention is deliberately broader than the current production model.  A feature
can be kept because it may help acceleration/regime sidecars, HMM experiments,
NSA prediction, Kalman/fusion logic, SA diagnostics or prediction, or another
local time-series model.  The policy is only meant to stop emitting families we
can defend as noise or compute drag.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALLOW_MANIFEST = REPO_ROOT / "usable_exact_features.csv"
DEFAULT_DENY_MANIFEST = REPO_ROOT / "non_interesting_exact_features.csv"

VALID_POLICY_MODES = frozenset({"off", "audit", "usable_exact", "strict"})
RETENTION_USE_CASES = frozenset(
    {
        "acceleration",
        "hmm",
        "nsa_prediction",
        "kalman",
        "sa_prediction",
        "other_time_series",
    }
)
METADATA_COLUMNS = frozenset(
    {
        "date",
        "ds",
        "release_date",
        "snapshot_date",
        "snapshot_month",
        "series_code",
    }
)

FAMILY_POLICY_VERSION = "2026-05-16-feature-generation-cull-v1"

PROTECTED_EXACT_FEATURES = frozenset(
    {
        "NFP_Consensus_Mean",
        "NFP_Consensus_Mean_diff",
        "rev_master_mean",
        "nfp_nsa_accel_lag6",
        "nfp_nsa_accel_vol_3m",
        "nfp_nsa_mom_abs_lag1",
        "nfp_nsa_mom_vol_3m",
        "nfp_nsa_mom_lag6",
        "nfp_nsa_mom_lag12",
        "sanagap_adj_lag1",
    }
)

PROTECTED_PREFIXES = (
    "NFP_Consensus",
    "rev_master_",
    "nfp_nsa_",
    "sanagap_adj_",
)

ZERO_CROSSING_PCT_TOKENS = (
    "rate",
    "spread",
    "stress",
    "vol",
    "volatility",
    "vix",
    "yield_curve",
    "treasury",
    "fedfund",
    "drawdown",
    "acceleration",
    "diffusion",
)


@dataclass(frozen=True)
class FeatureDecision:
    """Decision for one emitted feature name."""

    emit: bool
    decision: str
    reason: str
    mode: str


_SANITIZE_MULTI_CHAR = {
    "%": "pct",
    "&": "and",
    "+": "plus",
    "<": "lt",
    ">": "gt",
}
_SANITIZE_MULTI_RE = re.compile("|".join(re.escape(k) for k in _SANITIZE_MULTI_CHAR))
_SANITIZE_TO_UNDERSCORE = re.compile(r"[|\s\[\]{}\\,()\?/:;!@#$*=.<>]")
_SANITIZE_STRIP_QUOTES = re.compile(r"[\"']")
_SANITIZE_INTERIOR_HYPHEN = re.compile(r"(?<!^)-(?!$)")
_SANITIZE_COLLAPSE = re.compile(r"_+")


def sanitize_feature_name(name: str) -> str:
    """Mirror Train.data_loader.sanitize_feature_name without importing Train."""
    name = str(name)
    name = _SANITIZE_MULTI_RE.sub(lambda m: _SANITIZE_MULTI_CHAR[m.group()], name)
    name = _SANITIZE_STRIP_QUOTES.sub("", name)
    name = _SANITIZE_INTERIOR_HYPHEN.sub("_", name)
    name = _SANITIZE_TO_UNDERSCORE.sub("_", name)
    return _SANITIZE_COLLAPSE.sub("_", name).strip("_")


def current_feature_policy_mode() -> str:
    mode = os.getenv("NFP_FEATURE_POLICY", "off").strip().lower()
    if mode not in VALID_POLICY_MODES:
        return "off"
    return mode


def _manifest_path(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path)


def allow_manifest_path() -> Path:
    return _manifest_path("NFP_FEATURE_POLICY_ALLOW", DEFAULT_ALLOW_MANIFEST)


def deny_manifest_path() -> Path:
    return _manifest_path("NFP_FEATURE_POLICY_DENY", DEFAULT_DENY_MANIFEST)


def _file_digest(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def feature_policy_schema_version() -> str:
    return "::".join(
        [
            FAMILY_POLICY_VERSION,
            f"allow={_file_digest(allow_manifest_path())}",
            f"deny={_file_digest(deny_manifest_path())}",
        ]
    )


@lru_cache(maxsize=16)
def _load_manifest_map(path_key: str) -> dict[str, frozenset[str]]:
    path = Path(path_key)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, usecols=["source", "feature"], low_memory=False)
    except ValueError:
        return {}
    if frame.empty:
        return {}

    frame = frame.dropna(subset=["source", "feature"])
    by_source: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        source = str(row.source)
        feature = str(row.feature)
        by_source[source].add(feature)
        by_source[source].add(sanitize_feature_name(feature))
    return {source: frozenset(values) for source, values in by_source.items()}


def exact_allow_map() -> dict[str, frozenset[str]]:
    return _load_manifest_map(str(allow_manifest_path().resolve()))


def exact_deny_map() -> dict[str, frozenset[str]]:
    return _load_manifest_map(str(deny_manifest_path().resolve()))


def exact_allowed_features(source_name: str) -> frozenset[str]:
    return exact_allow_map().get(str(source_name), frozenset())


def exact_denied_features(source_name: str) -> frozenset[str]:
    return exact_deny_map().get(str(source_name), frozenset())


def _is_exact_protected(feature_name: str) -> bool:
    feature = str(feature_name)
    return feature in PROTECTED_EXACT_FEATURES or any(
        feature.startswith(prefix) for prefix in PROTECTED_PREFIXES
    )


def _has_exact_allowed_descendant(source_name: str, feature_name: str) -> bool:
    feature = str(feature_name)
    allowed = exact_allowed_features(source_name)
    if not allowed:
        return False
    prefix = f"{feature}_"
    return any(name == feature or name.startswith(prefix) for name in allowed)


def _family_block_reason(source_name: str, feature_name: str) -> str | None:
    source = str(source_name or "Unknown")
    feature = str(feature_name)
    lower = feature.lower()

    if source == "Unknown":
        return "unknown_source"

    if source == "NOAA":
        return "blocked_noaa_by_default"

    if source == "Prosper":
        return "blocked_prosper_by_default"

    if "_symlog" in lower:
        return "blocked_symlog_family"

    if "_diff_zscore_12m" in lower:
        return "blocked_12m_diff_zscore"

    if "_zscore_" in lower and "_diff_zscore_" not in lower:
        return "blocked_level_zscore"

    if "_pct_chg" in lower and any(token in lower for token in ZERO_CROSSING_PCT_TOKENS):
        return "blocked_unstable_pct_change"

    if re.search(r"_lag_(18|24|36)m$", lower):
        return "blocked_deep_lag"

    if re.search(r"_rolling_(mean|std)_(12|18|24|36)m", lower):
        return "blocked_deep_rolling"

    return None


_DECISION_COUNTS: Counter[tuple[str, str, str, str]] = Counter()
_DECISION_SAMPLES: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)


def _record_decision(source_name: str, feature_name: str, decision: FeatureDecision) -> None:
    if decision.mode == "off":
        return
    key = (decision.mode, str(source_name or "Unknown"), decision.decision, decision.reason)
    _DECISION_COUNTS[key] += 1
    samples = _DECISION_SAMPLES[key]
    if len(samples) < 10:
        samples.append(str(feature_name))


def reset_feature_policy_audit() -> None:
    _DECISION_COUNTS.clear()
    _DECISION_SAMPLES.clear()


def should_emit_feature(
    source_name: str | None,
    feature_name: str,
    *,
    mode: str | None = None,
    record: bool = True,
) -> FeatureDecision:
    """Return the final-emission decision for a feature name."""
    active_mode = (mode or current_feature_policy_mode()).strip().lower()
    if active_mode not in VALID_POLICY_MODES:
        active_mode = "off"

    source = str(source_name or "Unknown")
    feature = str(feature_name)

    if active_mode == "off":
        decision = FeatureDecision(True, "allow", "policy_off", active_mode)
    elif feature in exact_denied_features(source):
        emit = active_mode == "audit"
        decision = FeatureDecision(emit, "audit_deny" if emit else "deny", "exact_deny", active_mode)
    elif feature in exact_allowed_features(source):
        decision = FeatureDecision(True, "allow", "exact_allow", active_mode)
    elif _is_exact_protected(feature):
        decision = FeatureDecision(True, "protect", "protected_recurring_signal", active_mode)
    elif active_mode == "strict":
        decision = FeatureDecision(False, "deny", "strict_not_allowlisted", active_mode)
    else:
        reason = _family_block_reason(source, feature)
        if reason is None:
            decision = FeatureDecision(True, "allow", "not_blocked", active_mode)
        else:
            emit = active_mode == "audit"
            decision = FeatureDecision(
                emit,
                "audit_deny" if emit else "deny",
                reason,
                active_mode,
            )

    if record:
        _record_decision(source, feature, decision)
    return decision


def should_generate_intermediate_feature(source_name: str | None, feature_name: str) -> bool:
    """Return whether an intermediate transform should be materialized."""
    source = str(source_name or "Unknown")
    feature = str(feature_name)
    mode = current_feature_policy_mode()
    if mode in {"off", "audit"}:
        return True

    decision = should_emit_feature(source, feature, mode=mode, record=False)
    if decision.emit:
        return True

    return _has_exact_allowed_descendant(source, feature) or _is_exact_protected(feature)


def filter_feature_names(
    source_name: str | None,
    feature_names: Iterable[str],
    *,
    mode: str | None = None,
) -> list[str]:
    """Filter feature names according to the active generation policy."""
    out: list[str] = []
    for name in feature_names:
        decision = should_emit_feature(source_name, str(name), mode=mode)
        if decision.emit:
            out.append(str(name))
    return out


def filter_long_features(
    df: pd.DataFrame,
    source_name: str | None,
    *,
    feature_col: str = "series_name",
    mode: str | None = None,
) -> pd.DataFrame:
    if df.empty or feature_col not in df.columns:
        return df
    names = df[feature_col].astype(str).drop_duplicates().tolist()
    keep = set(filter_feature_names(source_name, names, mode=mode))
    active_mode = (mode or current_feature_policy_mode()).strip().lower()
    if active_mode == "audit" or len(keep) == len(names):
        return df
    return df.loc[df[feature_col].astype(str).isin(keep)].copy()


def filter_wide_features(
    df: pd.DataFrame,
    source_name: str | None,
    *,
    metadata_columns: Iterable[str] = METADATA_COLUMNS,
    mode: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    protected = {str(c) for c in metadata_columns}
    feature_cols = [str(c) for c in df.columns if str(c) not in protected]
    keep = set(filter_feature_names(source_name, feature_cols, mode=mode))
    active_mode = (mode or current_feature_policy_mode()).strip().lower()
    if active_mode == "audit" or len(keep) == len(feature_cols):
        return df
    keep_cols = [c for c in df.columns if str(c) in protected or str(c) in keep]
    return df.loc[:, keep_cols].copy()


def feature_transform_metadata(feature_name: str) -> dict[str, str]:
    feature = str(feature_name)
    lag = "none"
    base = feature
    for suffix in ("_lag_12m", "_lag_6m", "_lag_3m", "_lag_1m"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            lag = suffix[1:]
            break

    transform_family = "raw"
    for token, family in (
        ("_diff_zscore_12m", "diff_zscore_12m"),
        ("_diff_zscore_3m", "diff_zscore_3m"),
        ("_rolling_mean_3m", "rolling_mean_3m"),
        ("_rolling_std_6m", "rolling_std_6m"),
        ("_zscore_12m", "level_zscore_12m"),
        ("_zscore_3m", "level_zscore_3m"),
        ("_chg_12m", "chg_12m"),
        ("_chg_6m", "chg_6m"),
        ("_chg_3m", "chg_3m"),
        ("_pct_chg", "pct_chg"),
        ("_diff", "diff"),
        ("_symlog", "symlog"),
    ):
        if base.endswith(token):
            base = base[: -len(token)]
            transform_family = family
            break

    return {
        "base_series": base,
        "transform_family": transform_family,
        "lag": lag,
    }


def write_feature_policy_report(report_dir: str | Path | None = None) -> Path | None:
    """Write the accumulated feature-culling summary and return the CSV path."""
    if not _DECISION_COUNTS:
        return None

    if report_dir is None:
        raw = os.getenv("NFP_FEATURE_POLICY_REPORT_DIR", "").strip()
        if raw:
            report_root = Path(raw)
        else:
            output_dir = Path(os.getenv("OUTPUT_DIR", "_output"))
            report_root = output_dir / "feature_culling"
    else:
        report_root = Path(report_dir)
    if not report_root.is_absolute():
        report_root = REPO_ROOT / report_root
    report_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, count in sorted(_DECISION_COUNTS.items()):
        mode, source, decision, reason = key
        rows.append(
            {
                "mode": mode,
                "source": source,
                "decision": decision,
                "reason": reason,
                "count": int(count),
                "sample_features": "|".join(_DECISION_SAMPLES.get(key, [])),
                "policy_schema_version": feature_policy_schema_version(),
            }
        )
    out = pd.DataFrame(rows)
    path = report_root / "feature_generation_policy_summary.csv"
    out.to_csv(path, index=False)
    return path
