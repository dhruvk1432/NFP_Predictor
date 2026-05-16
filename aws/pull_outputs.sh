#!/usr/bin/env bash
# Sync s3://<bucket>/_output/ -> local ./_output

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight
ensure_bucket_name

LOCAL="${REPO_ROOT}/_output"
mkdir -p "${LOCAL}"
SRC="s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}/"
log "Syncing ${SRC} -> ${LOCAL}"
aws_ s3 sync "${SRC}" "${LOCAL}/"
log "Done."
