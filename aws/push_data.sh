#!/usr/bin/env bash
# Sync ./data/<subdir>/ -> s3://<bucket>/data/<subdir>/ for the minimal
# train-only dataset. Current train-all reads PIT master snapshots plus the
# revised NSA/SA target parquets; raw/prepared ETL inputs are not required.
#
# Whitelist (per Train/config.py):
#   - master_snapshots/  (~17G) — the wide-format monthly PIT snapshots
#   - NFP_target/        (~152K) — y_nsa_revised / y_sa_revised target series
#
# Skipped (used only by Data_ETA_Pipeline/ or legacy experiments, not current
# train-all from existing snapshots):
#   - fred_data/, Exogenous_data/, fred_data_prepared_nsa/,
#     fred_data_prepared_sa/
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
  log "PUSH_DATA_ALL=1, syncing every data/ subdir for ETL-capable AWS."
else
  SUBDIRS=(master_snapshots NFP_target)
  log "Syncing train-only whitelist. Set PUSH_DATA_ALL=1 to push ETL inputs too."
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
    --delete \
    --exclude "*.tmp" --exclude "*.DS_Store" --exclude "__pycache__/*"
done

log "Done."
