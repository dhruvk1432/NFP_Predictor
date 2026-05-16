#!/usr/bin/env bash
# Runs ON the EC2 instance. Syncs ./_output -> s3://<bucket>/_output

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"
log() { printf '[push-out %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

BUCKET="${S3_BUCKET:-}"
if [[ -z "${BUCKET}" ]]; then
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"
  [[ -n "${ACCOUNT_ID}" ]] || { echo "ERROR: cannot resolve AWS account id"; exit 1; }
  BUCKET="nfp-predictor-${ACCOUNT_ID}"
fi

[[ -d "${PROJECT_DIR}/_output" ]] || { echo "No _output to push."; exit 0; }
log "Syncing ${PROJECT_DIR}/_output/ -> s3://${BUCKET}/_output/"
aws s3 sync "${PROJECT_DIR}/_output/" "s3://${BUCKET}/_output/" \
  --exclude "*.tmp" --exclude ".DS_Store"
log "Done."
