#!/usr/bin/env bash
# Fire-and-forget training launcher.
#
# Usage:
#   ./aws/run_training.sh                          # default: python Train/train_lightgbm_nfp.py --train-all
#   ./aws/run_training.sh --train-all              # explicit (same as above)
#   ./aws/run_training.sh --branch nsa_first       # forward any flags to train_lightgbm_nfp.py
#   NO_AUTO_STOP=1 ./aws/run_training.sh           # don't auto-stop the instance when done
#
# What it does:
#   1. starts the EC2 instance if stopped
#   2. rsyncs your local code (Train/, Data_ETA_Pipeline/, etc.)
#   3. launches a detached tmux session called `train` that runs:
#        aws/on_instance/run_training.sh <your args>
#      which itself: trains -> pushes _output to S3 -> sudo shutdown -h +1
#   4. exits immediately. You can close your laptop.
#
# After kickoff you can:
#   aws/tail_log.sh                # tail the live log
#   aws/ssh.sh                     # then: tmux a -t train  (reattach to see live stdout)
#   aws/pull_outputs.sh            # once it's done, pull _output to your laptop

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight

# ---------------- ensure instance is up ------------------------------------
if ! [[ -f "${INSTANCE_ID_FILE}" ]]; then
  die "No instance provisioned. Run aws/provision.sh first."
fi
STATE="$(instance_state)"
if [[ "${STATE}" != "running" ]]; then
  log "Instance is '${STATE}'. Starting ..."
  "${AWS_DIR}/start.sh" >/dev/null
fi

# wait for SSH
log "Waiting for SSH ..."
until ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 -o BatchMode=yes "$(ssh_target)" 'echo ok' \
    2>/dev/null | grep -q ok; do
  sleep 5
done

# ---------------- push latest code -----------------------------------------
log "Rsyncing latest code to instance ..."
"${AWS_DIR}/push_code.sh" >/dev/null

# ---------------- launch ----------------------------------------------------
TARGET="$(ssh_target)"
# Build a single command string to run inside the detached tmux session.
# Note: we pass NO_AUTO_STOP through if it's set, and forward all our args.
NO_AUTO_STOP_VAL="${NO_AUTO_STOP:-0}"
REMOTE_CMD="cd ${REMOTE_PROJECT_DIR} && NO_AUTO_STOP=${NO_AUTO_STOP_VAL} bash aws/on_instance/run_training.sh"
for arg in "$@"; do
  # shell-quote each arg
  REMOTE_CMD+=" $(printf '%q' "$arg")"
done

log "Launching tmux session 'train' on $(instance_public_ip) ..."
QUOTED_REMOTE_CMD="$(printf '%q' "${REMOTE_CMD}")"
# Kill any prior 'train' session before starting a new one (idempotent).
ssh "${SSH_OPTS[@]}" "${TARGET}" \
  "tmux kill-session -t train 2>/dev/null; tmux new-session -d -s train ${QUOTED_REMOTE_CMD}"

log ""
log "Training launched. You can close your laptop."
log ""
log "  Monitor live log : aws/tail_log.sh"
log "  Reattach tmux    : aws/ssh.sh   then: tmux a -t train"
log "  Pull outputs     : aws/pull_outputs.sh   (after instance stops)"
log ""
if [[ "${NO_AUTO_STOP_VAL}" == "1" ]]; then
  log "NO_AUTO_STOP=1 set, instance will keep running. Remember aws/stop.sh."
else
  log "When training finishes, the instance will:"
  log "  1. sync _output/ to s3://${S3_BUCKET:-<bucket>}/_output/"
  log "  2. sudo shutdown -h +1 (auto-stop, EBS preserved)"
fi
