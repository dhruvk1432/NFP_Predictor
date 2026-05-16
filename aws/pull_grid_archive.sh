#!/usr/bin/env bash
# Mirror s3://<bucket>/_output_grid/ → ./_output_grid_archive/ on the laptop.
#
# This is your durability backstop: every cell that finishes on the instance
# is pushed to S3 immediately by Train/grid_search.py, and this script pulls
# from S3 to a local archive directory.
#
# Safe to run any time, while the grid is running or after. It NEVER deletes —
# `aws s3 sync` only adds new/changed objects unless --delete is passed.
#
# Cron-style usage (once an hour while grid is running):
#   while true; do ./aws/pull_grid_archive.sh && sleep 3600; done

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight
ensure_bucket_name

LOCAL="${REPO_ROOT}/_output_grid_archive"
mkdir -p "${LOCAL}"
SRC="s3://${S3_BUCKET}/_output_grid/"

log "Mirroring ${SRC} -> ${LOCAL}/"
log "  (additive only; no --delete; existing files preserved)"
aws_ s3 sync "${SRC}" "${LOCAL}/" \
  --exclude "*.tmp" --exclude ".DS_Store"

echo
echo "Local archive summary:"
echo "  total size : $(du -sh "${LOCAL}" 2>/dev/null | cut -f1)"
echo "  cells      : $(ls -1d "${LOCAL}"/cell_* 2>/dev/null | wc -l | tr -d ' ')"
echo "  results    : ${LOCAL}/grid_results.csv"
[[ -f "${LOCAL}/grid_results.csv" ]] && echo "  result rows: $(($(wc -l < "${LOCAL}/grid_results.csv") - 1))"
