#!/usr/bin/env bash
# Runs ON the EC2 instance.
#
# Sequentially runs replay-mode Panel/Kalman train-all jobs. This is used when
# dynamic feature selections from an in-flight cadence (usually reselect36) are
# reused by slower cadences so they do not recompute expensive reselection.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

export BACKTEST_MONTHS="${BACKTEST_MONTHS:-197}"
export NFP_PANEL_ROUTER_SELECTION_LOOKBACK="${NFP_PANEL_ROUTER_SELECTION_LOOKBACK:-24}"
export NFP_TRAIN_FEATURE_N_JOBS="${NFP_TRAIN_FEATURE_N_JOBS:-5}"
export NFP_LGBM_N_JOBS="${NFP_LGBM_N_JOBS:-5}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NO_AUTO_STOP="${NO_AUTO_STOP:-1}"

SOURCE_OUTPUT_DIR="${PER_WINDOW_FEATURES_SOURCE_OUTPUT_DIR:-_output_panel_router_2010_reselect36}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_replay36}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
declare -a CADENCES=(${REPLAY_SEQUENCE_CADENCES:-100 60})

SEQ_OUTPUT_DIR="${SEQ_OUTPUT_DIR:-_output_panel_router_2010_replay_sequence}"
LOG_DIR="${PROJECT_DIR}/${SEQ_OUTPUT_DIR}/runs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TS}.log"

log() { printf '[panel-router-replay-seq %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"; }

log "Starting replay sequence"
log "CADENCES=${CADENCES[*]}"
log "SOURCE_OUTPUT_DIR=${SOURCE_OUTPUT_DIR}"
log "OUTPUT_SUFFIX=${OUTPUT_SUFFIX}"
log "BACKTEST_MONTHS=${BACKTEST_MONTHS}"
log "NFP_PANEL_ROUTER_SELECTION_LOOKBACK=${NFP_PANEL_ROUTER_SELECTION_LOOKBACK}"
log "NFP_TRAIN_FEATURE_N_JOBS=${NFP_TRAIN_FEATURE_N_JOBS}"
log "NFP_LGBM_N_JOBS=${NFP_LGBM_N_JOBS}"
log "RESELECTION_EXCLUDE_ANCHOR=1"
log "PER_WINDOW_FEATURES_EXCLUDE_ANCHOR=1"

if [[ -n "${WAIT_FOR_PID}" ]]; then
  log "Waiting for pid ${WAIT_FOR_PID} before starting sequence"
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep "${WAIT_POLL_SECONDS}"
  done
  log "Wait pid ${WAIT_FOR_PID} exited; starting sequence"
fi

SEQ_EXIT=0
printf 'cadence,status,started_at,finished_at,output_dir\n' > "${LOG_DIR}/statuses.csv"

for cadence in "${CADENCES[@]}"; do
  out_dir="_output_panel_router_2010_reselect${cadence}${OUTPUT_SUFFIX}"
  started_at="$(date -Iseconds)"
  log "Launching reselect${cadence}: OUTPUT_DIR=${out_dir}, source=${SOURCE_OUTPUT_DIR}"

  (
    export OUTPUT_DIR="${out_dir}"
    export TEMP_DIR="${out_dir}/temp"
    export RESELECT_EVERY_N_MONTHS="${cadence}"
    export RESELECTION_EXCLUDE_ANCHOR=1
    export USE_PER_WINDOW_FEATURES=True
    export PER_WINDOW_FEATURES_SOURCE_OUTPUT_DIR="${SOURCE_OUTPUT_DIR}"
    export PER_WINDOW_FEATURES_REPLAY_MODE=cadence
    export PER_WINDOW_FEATURES_EXCLUDE_ANCHOR=1
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
  ) >> "${LOG_FILE}" 2>&1

  status=$?
  finished_at="$(date -Iseconds)"
  printf '%s,%s,%s,%s,%s\n' "${cadence}" "${status}" "${started_at}" "${finished_at}" "${out_dir}" >> "${LOG_DIR}/statuses.csv"
  if [[ "${status}" -ne 0 ]]; then
    log "ERROR: reselect${cadence} failed with status ${status}; stopping sequence"
    SEQ_EXIT="${status}"
    break
  fi
  log "reselect${cadence} finished successfully"
done

log "Replay sequence done. Final status=${SEQ_EXIT}"
exit "${SEQ_EXIT}"
