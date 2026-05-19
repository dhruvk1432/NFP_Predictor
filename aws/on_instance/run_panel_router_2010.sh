#!/usr/bin/env bash
# Runs ON the EC2 instance.
#
# Isolated long-history train-all for the Panel/Kalman router work. This keeps
# the normal _output/ tree untouched by default and writes artifacts to
# _output_panel_router_2010/.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

export OUTPUT_DIR="${OUTPUT_DIR:-_output_panel_router_2010}"
export TEMP_DIR="${TEMP_DIR:-${OUTPUT_DIR}/temp}"
export BACKTEST_MONTHS="${BACKTEST_MONTHS:-197}"
export NFP_PANEL_ROUTER_SELECTION_LOOKBACK="${NFP_PANEL_ROUTER_SELECTION_LOOKBACK:-24}"
export NFP_TRAIN_FEATURE_N_JOBS="${NFP_TRAIN_FEATURE_N_JOBS:-5}"
export NFP_LGBM_N_JOBS="${NFP_LGBM_N_JOBS:-5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NFP_LGBM_N_JOBS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${NFP_LGBM_N_JOBS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NFP_LGBM_N_JOBS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${NFP_LGBM_N_JOBS}}"
export NFP_TRAIN_FEATURE_BACKEND="${NFP_TRAIN_FEATURE_BACKEND:-threading}"
export NFP_TRAIN_FEATURE_USE_SNAPSHOT_CACHE="${NFP_TRAIN_FEATURE_USE_SNAPSHOT_CACHE:-auto}"
export NFP_TRAIN_DATASET_CACHE_READ_ROOTS="${NFP_TRAIN_DATASET_CACHE_READ_ROOTS:-_output_pairing_baseline_pitfix}"

LOG_DIR="${PROJECT_DIR}/${OUTPUT_DIR}/runs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TS}.log"

log() { printf '[panel-router-2010 %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"; }

log "Starting isolated train-all"
log "OUTPUT_DIR=${OUTPUT_DIR}"
log "TEMP_DIR=${TEMP_DIR}"
log "BACKTEST_MONTHS=${BACKTEST_MONTHS}"
log "RESELECT_EVERY_N_MONTHS=${RESELECT_EVERY_N_MONTHS:-<settings default>}"
log "RESELECTION_EXCLUDE_ANCHOR=${RESELECTION_EXCLUDE_ANCHOR:-<settings default>}"
log "USE_PER_WINDOW_FEATURES=${USE_PER_WINDOW_FEATURES:-<settings default>}"
log "PER_WINDOW_FEATURES_SOURCE_OUTPUT_DIR=${PER_WINDOW_FEATURES_SOURCE_OUTPUT_DIR:-<none>}"
log "PER_WINDOW_FEATURES_REPLAY_MODE=${PER_WINDOW_FEATURES_REPLAY_MODE:-<settings default>}"
log "NFP_PANEL_ROUTER_SELECTION_LOOKBACK=${NFP_PANEL_ROUTER_SELECTION_LOOKBACK}"
log "NFP_TRAIN_FEATURE_N_JOBS=${NFP_TRAIN_FEATURE_N_JOBS}"
log "NFP_LGBM_N_JOBS=${NFP_LGBM_N_JOBS}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/.venv/bin/activate"
log "Python: $(python --version 2>&1) at $(which python)"

python Train/train_lightgbm_nfp.py --train-all 2>&1 | tee -a "${LOG_FILE}"
TRAIN_EXIT=${PIPESTATUS[0]}
log "Training exited with status ${TRAIN_EXIT}"

BUCKET="${S3_BUCKET:-}"
if [[ -z "${BUCKET}" ]]; then
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"
  if [[ -n "${ACCOUNT_ID}" ]]; then
    BUCKET="nfp-predictor-${ACCOUNT_ID}"
  fi
fi

if [[ -n "${BUCKET}" ]]; then
  log "Syncing ${OUTPUT_DIR}/ to s3://${BUCKET}/${OUTPUT_DIR}/"
  aws s3 sync "${OUTPUT_DIR}/" "s3://${BUCKET}/${OUTPUT_DIR}/" \
    --exclude "*.tmp" --exclude ".DS_Store" 2>&1 | tee -a "${LOG_FILE}" \
    || log "WARN: output sync failed"
else
  log "WARN: could not resolve S3 bucket; outputs remain on EBS only"
fi

if [[ "${NO_AUTO_STOP:-0}" == "1" ]]; then
  log "NO_AUTO_STOP=1, leaving instance running"
else
  log "Scheduling shutdown in 1 min. Cancel with: sudo shutdown -c"
  sudo /sbin/shutdown -h +1 \
    "NFP panel-router 2010 history run finished (exit=${TRAIN_EXIT}); auto-stopping." \
    2>&1 | tee -a "${LOG_FILE}" || log "WARN: shutdown command failed"
fi

log "Launcher done. Final training exit status: ${TRAIN_EXIT}"
exit "${TRAIN_EXIT}"
