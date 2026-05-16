#!/usr/bin/env bash
# Runs ON the EC2 instance. Wrapper around Train/train_lightgbm_nfp.py that:
#   1. activates the venv
#   2. logs all output to _output/runs/run_<ts>.log
#   3. runs training with whatever args were passed in (default: --train-all)
#   4. always pushes _output/ to S3 afterward (even if training fails)
#   5. unless NO_AUTO_STOP=1, schedules `sudo shutdown -h +1` so the instance
#      stops itself ~1 min after training finishes
#
# Designed to be launched detached from a tmux session by aws/run_training.sh.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/_output/runs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TS}.log"

log() { printf '[run %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"; }

log "Training launcher starting. Logging to ${LOG_FILE}"
log "Args: $*"
log "NO_AUTO_STOP=${NO_AUTO_STOP:-0}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/.venv/bin/activate"
log "Python: $(python --version 2>&1) at $(which python)"

# ---------------- training -------------------------------------------------
log "==== train_lightgbm_nfp.py ===="
TRAIN_ARGS=("$@")
[[ "${#TRAIN_ARGS[@]}" -eq 0 ]] && TRAIN_ARGS=(--train-all)
# Use stdbuf so tee flushes line-by-line (live log tail-able).
stdbuf -oL -eL python Train/train_lightgbm_nfp.py "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
TRAIN_EXIT=${PIPESTATUS[0]}
log "==== training exited with status ${TRAIN_EXIT} ===="

# ---------------- push outputs (always) ------------------------------------
log "==== syncing _output/ to S3 ===="
bash "${PROJECT_DIR}/aws/on_instance/push_outputs.sh" 2>&1 | tee -a "${LOG_FILE}" || \
  log "WARN: push_outputs.sh failed; outputs remain on this EBS volume."

# ---------------- auto-stop (default) --------------------------------------
if [[ "${NO_AUTO_STOP:-0}" == "1" ]]; then
  log "NO_AUTO_STOP=1, leaving instance running."
else
  log "Scheduling shutdown in 1 min. Cancel with: sudo shutdown -c"
  # EC2 InstanceInitiatedShutdownBehavior defaults to 'stop' (not terminate)
  # for instances launched via run-instances, so this preserves the EBS volume.
  sudo /sbin/shutdown -h +1 "NFP training finished (exit=${TRAIN_EXIT}); auto-stopping." \
    2>&1 | tee -a "${LOG_FILE}" || log "WARN: shutdown command failed."
fi

log "Launcher done. Final training exit status: ${TRAIN_EXIT}"
exit "${TRAIN_EXIT}"
