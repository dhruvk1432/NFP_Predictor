#!/usr/bin/env bash
# Runs ON the EC2 instance. Wrapper around Train/grid_search.py that:
#   1. activates the venv
#   2. installs any missing dependencies (jinja2, optuna-integration[lightgbm])
#   3. logs all output to _output_grid/grid_master_<ts>.log
#   4. runs grid_search.py
#   5. ALWAYS pushes _output_grid/ to S3 afterward (even if it fails)
#   6. unless NO_AUTO_STOP=1, schedules `sudo shutdown -h +1` so the instance
#      stops itself ~1 min after grid finishes
#
# Designed to be launched detached from a tmux session by aws/run_grid_search.sh.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/_output_grid"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/grid_master_${TS}.log"

log() { printf '[grid %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${LOG_FILE}"; }

log "Grid launcher starting. Logging to ${LOG_FILE}"
log "NO_AUTO_STOP=${NO_AUTO_STOP:-0}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/.venv/bin/activate"
log "Python: $(python --version 2>&1) at $(which python)"

# ---------------- dependency hardening -------------------------------------
# These two were the AWS-vs-local portability gaps that took two run attempts
# to bridge. Idempotent: pip install is a no-op if already present.
log "==== ensuring runtime deps (jinja2, optuna-integration[lightgbm]) ===="
python -m pip install --quiet 'jinja2>=3' 'optuna-integration[lightgbm]' 2>&1 | tee -a "${LOG_FILE}" || \
  log "WARN: pip install partial failure; continuing."

# ---------------- grid search ----------------------------------------------
log "==== Train/grid_search.py ===="
stdbuf -oL -eL python Train/grid_search.py 2>&1 | tee -a "${LOG_FILE}"
GRID_EXIT=${PIPESTATUS[0]}
log "==== grid exited with status ${GRID_EXIT} ===="

# ---------------- push outputs (always) ------------------------------------
log "==== syncing _output_grid/ to S3 ===="
BUCKET="${S3_BUCKET:-nfp-predictor-989571801493}"
aws s3 sync "${PROJECT_DIR}/_output_grid/" "s3://${BUCKET}/_output_grid/" \
  --exclude "*.tmp" --exclude ".DS_Store" 2>&1 | tee -a "${LOG_FILE}" || \
  log "WARN: _output_grid sync failed; outputs remain on EBS volume."

# Also push the regular _output for any incidental writes (e.g. caches).
log "==== syncing _output/ to S3 ===="
aws s3 sync "${PROJECT_DIR}/_output/" "s3://${BUCKET}/_output/" \
  --exclude "*.tmp" --exclude ".DS_Store" 2>&1 | tee -a "${LOG_FILE}" || \
  log "WARN: _output sync failed."

# ---------------- auto-stop (default) --------------------------------------
if [[ "${NO_AUTO_STOP:-0}" == "1" ]]; then
  log "NO_AUTO_STOP=1, leaving instance running."
else
  log "Scheduling shutdown in 1 min. Cancel with: sudo shutdown -c"
  sudo /sbin/shutdown -h +1 "NFP grid finished (exit=${GRID_EXIT}); auto-stopping." \
    2>&1 | tee -a "${LOG_FILE}" || log "WARN: shutdown command failed."
fi

log "Launcher done. Final grid exit status: ${GRID_EXIT}"
exit "${GRID_EXIT}"
