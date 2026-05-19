#!/usr/bin/env bash
# Runs ON the EC2 instance.
#
# Launches isolated long-history Panel/Kalman train-all backtests with a
# bounded parallel scheduler. The default split is 3 concurrent jobs x 5
# threads = 15 total worker threads on the 16-vCPU instance.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

export BACKTEST_MONTHS="${BACKTEST_MONTHS:-197}"
export NFP_PANEL_ROUTER_SELECTION_LOOKBACK="${NFP_PANEL_ROUTER_SELECTION_LOOKBACK:-24}"
export NFP_TRAIN_FEATURE_N_JOBS="${NFP_TRAIN_FEATURE_N_JOBS:-5}"
export NFP_LGBM_N_JOBS="${NFP_LGBM_N_JOBS:-5}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

GRID_OUTPUT_DIR="${GRID_OUTPUT_DIR:-_output_panel_router_2010_reselect_grid}"
LOG_DIR="${PROJECT_DIR}/${GRID_OUTPUT_DIR}/runs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TS}.log"

log() { printf '[panel-router-reselect-grid %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"; }

log "Starting parallel reselection grid"
log "BACKTEST_MONTHS=${BACKTEST_MONTHS}"
log "NFP_PANEL_ROUTER_SELECTION_LOOKBACK=${NFP_PANEL_ROUTER_SELECTION_LOOKBACK}"
log "Per-job NFP_TRAIN_FEATURE_N_JOBS=${NFP_TRAIN_FEATURE_N_JOBS}"
log "Per-job NFP_LGBM_N_JOBS=${NFP_LGBM_N_JOBS}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
declare -a CADENCES=(${RESELECT_GRID_CADENCES:-200})
log "RESELECT grid cadences: ${CADENCES[*]}"
log "MAX_PARALLEL=${MAX_PARALLEL}"
log "Expected worker-thread total at full occupancy: $((MAX_PARALLEL * NFP_LGBM_N_JOBS))"
log "RESELECTION_EXCLUDE_ANCHOR=${RESELECTION_EXCLUDE_ANCHOR:-<settings default>}"
log "USE_PER_WINDOW_FEATURES=${USE_PER_WINDOW_FEATURES:-<settings default>}"
log "PER_WINDOW_FEATURES_SOURCE_OUTPUT_DIR=${PER_WINDOW_FEATURES_SOURCE_OUTPUT_DIR:-<none>}"
log "PER_WINDOW_FEATURES_REPLAY_MODE=${PER_WINDOW_FEATURES_REPLAY_MODE:-<settings default>}"

declare -A PID_TO_LABEL=()
declare -A LABEL_TO_STATUS=()
declare -a RUNNING_PIDS=()

launch_one() {
  local cadence="$1"
  local label="reselect${cadence}"
  local out_dir="_output_panel_router_2010_${label}"

  log "Launching ${label}: OUTPUT_DIR=${out_dir}, RESELECT_EVERY_N_MONTHS=${cadence}"
  (
    export OUTPUT_DIR="${out_dir}"
    export TEMP_DIR="${out_dir}/temp"
    export RESELECT_EVERY_N_MONTHS="${cadence}"
    export RESELECTION_EXCLUDE_ANCHOR="${RESELECTION_EXCLUDE_ANCHOR:-}"
    export USE_PER_WINDOW_FEATURES="${USE_PER_WINDOW_FEATURES:-}"
    export PER_WINDOW_FEATURES_SOURCE_OUTPUT_DIR="${PER_WINDOW_FEATURES_SOURCE_OUTPUT_DIR:-}"
    export PER_WINDOW_FEATURES_REPLAY_MODE="${PER_WINDOW_FEATURES_REPLAY_MODE:-}"
    export PER_WINDOW_FEATURES_EXCLUDE_ANCHOR="${PER_WINDOW_FEATURES_EXCLUDE_ANCHOR:-}"
    export BACKTEST_MONTHS
    export NFP_PANEL_ROUTER_SELECTION_LOOKBACK
    export NFP_TRAIN_FEATURE_N_JOBS
    export NFP_LGBM_N_JOBS
    export OMP_NUM_THREADS="${NFP_LGBM_N_JOBS}"
    export OPENBLAS_NUM_THREADS="${NFP_LGBM_N_JOBS}"
    export MKL_NUM_THREADS="${NFP_LGBM_N_JOBS}"
    export NUMEXPR_NUM_THREADS="${NFP_LGBM_N_JOBS}"
    export PYTHONUNBUFFERED
    export NO_AUTO_STOP=1
    bash aws/on_instance/run_panel_router_2010.sh
  ) >> "${LOG_FILE}" 2>&1 &

  local pid=$!
  PID_TO_LABEL["${pid}"]="${label}"
  RUNNING_PIDS+=("${pid}")
  printf '%s\n' "${pid}" > "${LOG_DIR}/${label}.pid"
  log "Launched ${label} as pid ${pid}"
}

rebuild_running() {
  local -a still_running=()
  local pid
  for pid in "${RUNNING_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      still_running+=("${pid}")
    fi
  done
  RUNNING_PIDS=("${still_running[@]}")
}

wait_for_one_slot() {
  local finished_pid
  local status
  if wait -n -p finished_pid; then
    status=0
  else
    status=$?
  fi
  local label="${PID_TO_LABEL[${finished_pid}]:-unknown}"
  if [[ "${status}" -eq 0 ]]; then
    log "${label} finished successfully"
  else
    log "ERROR: ${label} failed with status ${status}"
    GRID_EXIT=1
  fi
  LABEL_TO_STATUS["${label}"]="${status}"
  printf '%s,%s\n' "${label}" "${status}" >> "${LOG_DIR}/statuses.csv"
  rebuild_running
}

GRID_EXIT=0
printf 'label,status\n' > "${LOG_DIR}/statuses.csv"

for cadence in "${CADENCES[@]}"; do
  while [[ "${#RUNNING_PIDS[@]}" -ge "${MAX_PARALLEL}" ]]; do
    wait_for_one_slot
  done
  launch_one "${cadence}"
done

while [[ "${#RUNNING_PIDS[@]}" -gt 0 ]]; do
  wait_for_one_slot
done

for cadence in "${CADENCES[@]}"; do
  label="reselect${cadence}"
  if [[ "${LABEL_TO_STATUS[${label}]:-missing}" != "0" ]]; then
    GRID_EXIT=1
  fi
done

BUCKET="${S3_BUCKET:-}"
if [[ -z "${BUCKET}" ]]; then
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"
  if [[ -n "${ACCOUNT_ID}" ]]; then
    BUCKET="nfp-predictor-${ACCOUNT_ID}"
  fi
fi

if [[ -n "${BUCKET}" ]]; then
  log "Syncing ${GRID_OUTPUT_DIR}/ to s3://${BUCKET}/${GRID_OUTPUT_DIR}/"
  aws s3 sync "${GRID_OUTPUT_DIR}/" "s3://${BUCKET}/${GRID_OUTPUT_DIR}/" \
    --exclude "*.tmp" --exclude ".DS_Store" 2>&1 | tee -a "${LOG_FILE}" \
    || log "WARN: grid log sync failed"
else
  log "WARN: could not resolve S3 bucket; grid logs remain on EBS only"
fi

if [[ "${NO_AUTO_STOP:-0}" == "1" ]]; then
  log "NO_AUTO_STOP=1, leaving instance running"
else
  log "Scheduling shutdown in 1 min. Cancel with: sudo shutdown -c"
  sudo /sbin/shutdown -h +1 \
    "NFP panel-router reselection grid finished (exit=${GRID_EXIT}); auto-stopping." \
    2>&1 | tee -a "${LOG_FILE}" || log "WARN: shutdown command failed"
fi

log "Grid launcher done. Final exit status: ${GRID_EXIT}"
exit "${GRID_EXIT}"
