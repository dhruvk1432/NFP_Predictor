#!/usr/bin/env bash
# Sync ./data/<subdir>/ -> s3://<bucket>/data/<subdir>/ for each subdir
# that the training pipeline actually reads.
#
# Whitelist (per Train/config.py):
#   - master_snapshots/  (~50G) — the wide-format monthly snapshots
#   - fred_data/         (~124M) — FRED employment series used for features
#   - NFP_target/        (~152K) — y_nsa_revised / y_sa_revised target series
#
# Skipped (used only by Data_ETA_Pipeline/, not by training):
#   - Exogenous_data/, fred_data_prepared_nsa/, fred_data_prepared_sa/
#
# To sync everything (for full ETL on AWS), set PUSH_DATA_ALL=1 in env.
#
# Resumable: rerun anytime; `aws s3 sync` only uploads new/changed files.

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight
ensure_bucket_name

LOCAL_DATA_DIR="${REPO_ROOT}/data"
[[ -d "${LOCAL_DATA_DIR}" ]] || die "No local data dir at ${LOCAL_DATA_DIR}"

# Bump parallelism for faster uploads. Tweak in config.local.env if needed.
aws_ configure set default.s3.max_concurrent_requests 32
aws_ configure set default.s3.multipart_chunksize 64MB
aws_ configure set default.s3.max_bandwidth ""  # remove cap

PUSH_DATA_ALL="${PUSH_DATA_ALL:-0}"
if [[ "${PUSH_DATA_ALL}" == "1" ]]; then
  SUBDIRS=(master_snapshots fred_data NFP_target \
           Exogenous_data fred_data_prepared_nsa fred_data_prepared_sa)
  log "PUSH_DATA_ALL=1, syncing every data/ subdir (~69GB)."
else
  SUBDIRS=(master_snapshots fred_data NFP_target)
  log "Syncing training-only whitelist (~50GB). Set PUSH_DATA_ALL=1 to push everything."
fi

for sub in "${SUBDIRS[@]}"; do
  SRC="${LOCAL_DATA_DIR}/${sub}/"
  DST="s3://${S3_BUCKET}/${S3_DATA_PREFIX}/${sub}/"
  if [[ ! -d "${SRC}" ]]; then
    log "  (skipping ${sub}: not present locally)"
    continue
  fi
  log "  ${sub}: $(du -sh "${SRC}" 2>/dev/null | cut -f1) -> ${DST}"
  aws_ s3 sync "${SRC}" "${DST}" \
    --exclude "*.tmp" --exclude "*.DS_Store" --exclude "__pycache__/*"
done

log "Done."
