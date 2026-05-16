"""Exact-feature allow/deny helpers for audited feature manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib

import pandas as pd

from utils.feature_generation_policy import (
    DEFAULT_ALLOW_MANIFEST,
    DEFAULT_DENY_MANIFEST,
    METADATA_COLUMNS,
    exact_allow_map,
    exact_deny_map,
    exact_denied_features,
    feature_policy_schema_version,
    filter_feature_names as policy_filter_feature_names,
)


WIDE_PROTECTED_COLUMNS = METADATA_COLUMNS


def exact_feature_prune_map(
    manifest_path: str | Path | None = None,
) -> dict[str, frozenset[str]]:
    """Return ``{source_name: {exact_feature_names_to_drop}}``."""
    if manifest_path is None:
        return exact_deny_map()
    path = Path(manifest_path)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, usecols=["source", "feature"], low_memory=False)
    except ValueError:
        return {}
    frame = frame.dropna(subset=["source", "feature"])
    return {
        str(source): frozenset(group["feature"].astype(str).tolist())
        for source, group in frame.groupby("source", sort=False)
    }


def exact_feature_allow_map(
    allow_manifest_path: str | Path | None = None,
) -> dict[str, frozenset[str]]:
    """Return ``{source_name: {exact_feature_names_to_keep}}``."""
    if allow_manifest_path is None:
        return exact_allow_map()
    path = Path(allow_manifest_path)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path, usecols=["source", "feature"], low_memory=False)
    except ValueError:
        return {}
    frame = frame.dropna(subset=["source", "feature"])
    return {
        str(source): frozenset(group["feature"].astype(str).tolist())
        for source, group in frame.groupby("source", sort=False)
    }


def source_prune_features(
    source_name: str,
    manifest_path: str | Path | None = None,
) -> frozenset[str]:
    return exact_feature_prune_map(manifest_path).get(str(source_name), frozenset())


def source_allowed_features(
    source_name: str,
    allow_manifest_path: str | Path | None = None,
) -> frozenset[str]:
    return exact_feature_allow_map(allow_manifest_path).get(str(source_name), frozenset())


def filter_feature_names(
    source_name: str,
    feature_names: Iterable[str],
    manifest_path: str | Path | None = None,
    allow_manifest_path: str | Path | None = None,
) -> list[str]:
    """Return audited feature names, preferring an allow manifest when supplied."""
    if manifest_path is None and allow_manifest_path is None:
        return policy_filter_feature_names(source_name, feature_names)

    allowed = (
        source_allowed_features(source_name, allow_manifest_path)
        if allow_manifest_path is not None
        else frozenset()
    )
    if allowed:
        return [str(name) for name in feature_names if str(name) in allowed]

    blocked = source_prune_features(source_name, manifest_path)
    if not blocked:
        return [str(name) for name in feature_names]
    return [str(name) for name in feature_names if str(name) not in blocked]


def exact_feature_prune_schema_version(
    manifest_path: str | Path | None = None,
    allow_manifest_path: str | Path | None = None,
) -> str:
    deny_path = Path(manifest_path) if manifest_path is not None else DEFAULT_DENY_MANIFEST
    allow_path = (
        Path(allow_manifest_path)
        if allow_manifest_path is not None
        else DEFAULT_ALLOW_MANIFEST
    )

    def _digest(path: Path) -> str:
        if not path.exists():
            return "missing"
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    if manifest_path is None and allow_manifest_path is None:
        return feature_policy_schema_version()
    return f"exact-feature-prune::allow={_digest(allow_path)}::deny={_digest(deny_path)}"


def drop_non_interesting_wide(
    df: pd.DataFrame,
    source_name: str,
    *,
    protected_columns: Iterable[str] = WIDE_PROTECTED_COLUMNS,
    manifest_path: str | Path | None = None,
    allow_manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Reduce a wide snapshot/panel to an audited production feature set."""
    _ = manifest_path, allow_manifest_path
    if df.empty:
        return df
    protected = {str(c) for c in protected_columns}
    feature_cols = [str(c) for c in df.columns if str(c) not in protected]
    keep = set(
        filter_feature_names(
            source_name,
            feature_cols,
            manifest_path=manifest_path,
            allow_manifest_path=allow_manifest_path,
        )
    )
    if len(keep) == len(feature_cols):
        return df
    keep_cols = [c for c in df.columns if str(c) in protected or str(c) in keep]
    return df.loc[:, keep_cols].copy()


def drop_non_interesting_long(
    df: pd.DataFrame,
    source_name: str,
    *,
    feature_col: str = "series_name",
    manifest_path: str | Path | None = None,
    allow_manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Reduce a long-format snapshot/panel to an audited production feature set."""
    _ = manifest_path, allow_manifest_path
    if df.empty or feature_col not in df.columns:
        return df
    feature_names = df[feature_col].astype(str).drop_duplicates().tolist()
    keep = set(
        filter_feature_names(
            source_name,
            feature_names,
            manifest_path=manifest_path,
            allow_manifest_path=allow_manifest_path,
        )
    )
    if len(keep) == len(feature_names):
        return df
    return df.loc[df[feature_col].astype(str).isin(keep)].copy()
