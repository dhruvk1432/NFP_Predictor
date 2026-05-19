"""Refresh Reuters/LSEG economist-level NFP poll history from Unifier.

This loader maintains the raw local economist-panel cache used by the
dynamic panel features and sidecars:

    economist_panel/contributors.parquet
    economist_panel/by_economist/<ident>.parquet

The important Unifier detail is that the contributor identity table must be
queried first. Its ``ident`` values are then used as keys for
``lseg_us_reuters_polls_contributors`` to retrieve each economist's complete
forecast time series, including the recent rows that are not exposed by the
older hand-built cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from unifier import unifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from settings import TEMP_DIR, UNIFIER_TOKEN, UNIFIER_USER, setup_logger  # noqa: E402


logger = setup_logger(__file__, TEMP_DIR)

CONTRIBUTORS_DATASET = "lseg_us_reuters_polls_contributors_ident"
CONTRIBUTORS_KEY = "M#USNFARECI"
ECONOMIST_DATASET = "lseg_us_reuters_polls_contributors"

ECONOMIST_PANEL_DIR = PROJECT_ROOT / "economist_panel"
BY_ECONOMIST_DIR = ECONOMIST_PANEL_DIR / "by_economist"
CONTRIBUTORS_PATH = ECONOMIST_PANEL_DIR / "contributors.parquet"
MANIFEST_PATH = ECONOMIST_PANEL_DIR / "update_manifest.json"

REQUIRED_CONTRIBUTOR_COLUMNS = ("name", "ident")


@dataclass(frozen=True)
class FetchResult:
    ident: str
    name: str
    path: Path | None
    rows: int
    min_timestamp: str | None
    max_timestamp: str | None
    max_first_release_date: str | None
    error: str | None = None


def _configure_unifier() -> None:
    unifier.user = UNIFIER_USER
    unifier.token = UNIFIER_TOKEN
    os.environ["UNIFIER_USER"] = UNIFIER_USER
    os.environ["UNIFIER_TOKEN"] = UNIFIER_TOKEN


def _safe_ident_filename(ident: str) -> str:
    """Match the existing local filename convention, e.g. US&000Y1O -> US_000Y1O."""
    return str(ident).replace("&", "_").replace("/", "_")


def _iso_or_none(value: object) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def fetch_contributors() -> pd.DataFrame:
    """Fetch and validate the current contributor identity table."""
    _configure_unifier()
    contributors = unifier.get_dataframe(
        name=CONTRIBUTORS_DATASET,
        key=CONTRIBUTORS_KEY,
    )
    missing = [c for c in REQUIRED_CONTRIBUTOR_COLUMNS if c not in contributors.columns]
    if missing:
        raise RuntimeError(
            f"Contributor table missing columns {missing}; "
            f"available columns: {list(contributors.columns)}"
        )

    contributors = contributors.loc[:, list(REQUIRED_CONTRIBUTOR_COLUMNS)].copy()
    contributors["name"] = contributors["name"].astype(str).str.strip()
    contributors["ident"] = contributors["ident"].astype(str).str.strip()
    contributors = contributors.replace({"": pd.NA}).dropna(subset=["name", "ident"])
    contributors = (
        contributors.drop_duplicates(subset=["ident"], keep="first")
        .sort_values(["name", "ident"])
        .reset_index(drop=True)
    )
    if contributors.empty:
        raise RuntimeError("Contributor table was empty after cleaning")
    return contributors


def _normalize_economist_frame(df: pd.DataFrame, ident: str, name: str) -> pd.DataFrame:
    out = df.copy()
    if "ident" not in out.columns:
        out["ident"] = ident
    if "name" not in out.columns:
        out["name"] = name
    out["ident"] = out["ident"].fillna(ident).astype(str)
    out["name"] = out["name"].fillna(name).astype(str)

    sort_cols = [c for c in ("timestamp", "first_release_date", "last_revision_date") if c in out.columns]
    for col in sort_cols:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out


def _fetch_one_with_retries(
    ident: str,
    name: str,
    *,
    retries: int,
    retry_sleep: float,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            df = unifier.get_dataframe(name=ECONOMIST_DATASET, key=ident)
            if df.empty and len(df.columns) == 0:
                raise RuntimeError("Unifier returned an empty frame with no schema")
            return _normalize_economist_frame(df, ident, name)
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(str(last_error))


def fetch_and_write_economist(
    ident: str,
    name: str,
    *,
    output_dir: Path,
    retries: int,
    retry_sleep: float,
    dry_run: bool = False,
) -> FetchResult:
    """Fetch one economist and atomically write the local parquet cache."""
    try:
        _configure_unifier()
        df = _fetch_one_with_retries(
            ident,
            name,
            retries=retries,
            retry_sleep=retry_sleep,
        )

        timestamp = (
            pd.to_datetime(df["timestamp"], errors="coerce")
            if "timestamp" in df.columns
            else pd.Series(dtype="datetime64[ns]")
        )
        first_release = (
            pd.to_datetime(df["first_release_date"], errors="coerce")
            if "first_release_date" in df.columns
            else pd.Series(dtype="datetime64[ns]")
        )

        path = output_dir / f"{_safe_ident_filename(ident)}.parquet"
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            df.to_parquet(tmp_path, index=False)
            tmp_path.replace(path)

        return FetchResult(
            ident=ident,
            name=name,
            path=path,
            rows=int(len(df)),
            min_timestamp=_iso_or_none(timestamp.min()),
            max_timestamp=_iso_or_none(timestamp.max()),
            max_first_release_date=_iso_or_none(first_release.max()),
        )
    except Exception as exc:  # pragma: no cover - network dependent
        return FetchResult(
            ident=ident,
            name=name,
            path=None,
            rows=0,
            min_timestamp=None,
            max_timestamp=None,
            max_first_release_date=None,
            error=str(exc),
        )


def refresh_economist_panel(
    *,
    max_workers: int = 6,
    retries: int = 2,
    retry_sleep: float = 1.5,
    limit: int | None = None,
    idents: Iterable[str] | None = None,
    dry_run: bool = False,
) -> list[FetchResult]:
    contributors = fetch_contributors()
    if idents is not None:
        ident_set = set(idents)
        contributors = contributors[contributors["ident"].isin(ident_set)].reset_index(drop=True)
        missing = sorted(ident_set - set(contributors["ident"]))
        if missing:
            raise RuntimeError(f"Requested idents not present in contributor table: {missing}")
    if limit is not None:
        contributors = contributors.head(limit).copy()

    logger.info("Fetched %d contributor identities", len(contributors))
    ECONOMIST_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        contributors.to_parquet(CONTRIBUTORS_PATH, index=False)
        contributors.to_csv(CONTRIBUTORS_PATH.with_suffix(".csv"), index=False)
        logger.info("Wrote %s and CSV mirror", CONTRIBUTORS_PATH)

    results: list[FetchResult] = []
    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                fetch_and_write_economist,
                row.ident,
                row.name,
                output_dir=BY_ECONOMIST_DIR,
                retries=retries,
                retry_sleep=retry_sleep,
                dry_run=dry_run,
            ): (row.ident, row.name)
            for row in contributors.itertuples(index=False)
        }
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result.error:
                logger.error(
                    "[%d/%d] %s (%s) failed: %s",
                    i,
                    len(futures),
                    result.name,
                    result.ident,
                    result.error,
                )
            else:
                logger.info(
                    "[%d/%d] %s (%s): %d rows, max timestamp=%s, max first_release=%s",
                    i,
                    len(futures),
                    result.name,
                    result.ident,
                    result.rows,
                    result.max_timestamp,
                    result.max_first_release_date,
                )

    results.sort(key=lambda r: (r.name, r.ident))
    failures = [r for r in results if r.error]
    successes = [r for r in results if not r.error]
    manifest = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "contributors_dataset": CONTRIBUTORS_DATASET,
        "contributors_key": CONTRIBUTORS_KEY,
        "economist_dataset": ECONOMIST_DATASET,
        "contributors": int(len(contributors)),
        "successful": int(len(successes)),
        "failed": int(len(failures)),
        "max_timestamp": max((r.max_timestamp for r in successes if r.max_timestamp), default=None),
        "max_first_release_date": max(
            (r.max_first_release_date for r in successes if r.max_first_release_date),
            default=None,
        ),
        "failures": [
            {"ident": r.ident, "name": r.name, "error": r.error}
            for r in failures
        ],
    }
    if not dry_run:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
        logger.info("Wrote %s", MANIFEST_PATH)

    logger.info(
        "Economist panel refresh complete: %d successful, %d failed, max timestamp=%s",
        len(successes),
        len(failures),
        manifest["max_timestamp"],
    )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None, help="Fetch only the first N contributors")
    parser.add_argument(
        "--ident",
        action="append",
        dest="idents",
        help="Fetch only this ident; can be supplied multiple times",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit 0 even if one or more economist fetches fail",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = refresh_economist_panel(
        max_workers=args.max_workers,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        limit=args.limit,
        idents=args.idents,
        dry_run=args.dry_run,
    )
    failures = [r for r in results if r.error]
    if failures and not args.allow_failures:
        raise SystemExit(f"{len(failures)} economist fetches failed; see {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
