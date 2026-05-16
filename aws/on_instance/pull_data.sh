#!/usr/bin/env bash
# Runs ON the EC2 instance. Syncs s3://<bucket>/data -> ./data
# Bucket name is read from the instance's S3 access (works because of the
# IAM role attached at provision time).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"
log() { printf '[pull-data %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Derive bucket name from the account id (sts:GetCallerIdentity is always
# permitted; s3:ListAllMyBuckets is NOT in the instance role).
BUCKET="${S3_BUCKET:-}"
if [[ -z "${BUCKET}" ]]; then
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"
  [[ -n "${ACCOUNT_ID}" ]] || { echo "ERROR: cannot resolve AWS account id"; exit 1; }
  BUCKET="nfp-predictor-${ACCOUNT_ID}"
fi
log "Using bucket s3://${BUCKET}"

mkdir -p "${PROJECT_DIR}/data"
log "Syncing s3://${BUCKET}/data/ -> ${PROJECT_DIR}/data/"
aws s3 sync "s3://${BUCKET}/data/" "${PROJECT_DIR}/data/"
log "Done. data/ is now $(du -sh "${PROJECT_DIR}/data" | cut -f1)"
