#!/usr/bin/env bash
# Fire-and-forget grid-search launcher.
#
# Usage:
#   ./aws/run_grid_search.sh                       # launches Train/grid_search.py
#   NO_AUTO_STOP=1 ./aws/run_grid_search.sh        # don't auto-stop the instance when done
#
# What it does:
#   1. starts the EC2 instance if stopped
#   2. rsyncs your local code (Train/, Data_ETA_Pipeline/, etc.)
#   3. launches a detached tmux session called `grid` that runs:
#        aws/on_instance/run_grid_search.sh
#      which itself: installs deps -> runs grid -> pushes _output_grid/ to S3 -> sudo shutdown -h +1
#   4. exits immediately. You can close your laptop.
#
# After kickoff you can:
#   aws/ssh.sh                          # then: tmux a -t grid  (reattach to see live stdout)
#   aws/pull_grid.sh                    # once it's done, pull _output_grid/ to your laptop
#                                         (or use plain aws/pull_outputs.sh if added there)

set -euo pipefail
source "$(dirname "$0")/lib.sh"
preflight

if ! [[ -f "${INSTANCE_ID_FILE}" ]]; then
  die "No instance provisioned. Run aws/provision.sh first."
fi
STATE="$(instance_state)"
if [[ "${STATE}" != "running" ]]; then
  log "Instance is '${STATE}'. Starting ..."
  "${AWS_DIR}/start.sh" >/dev/null
fi

log "Waiting for SSH ..."
until ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 -o BatchMode=yes "$(ssh_target)" 'echo ok' \
    2>/dev/null | grep -q ok; do
  sleep 5
done

log "Rsyncing latest code to instance ..."
"${AWS_DIR}/push_code.sh" >/dev/null

TARGET="$(ssh_target)"
NO_AUTO_STOP_VAL="${NO_AUTO_STOP:-0}"
REMOTE_CMD="cd ${REMOTE_PROJECT_DIR} && NO_AUTO_STOP=${NO_AUTO_STOP_VAL} bash aws/on_instance/run_grid_search.sh"

log "Launching tmux session 'grid' on $(instance_public_ip) ..."
QUOTED_REMOTE_CMD="$(printf '%q' "${REMOTE_CMD}")"
ssh "${SSH_OPTS[@]}" "${TARGET}" \
  "tmux kill-session -t grid 2>/dev/null; tmux new-session -d -s grid ${QUOTED_REMOTE_CMD}"

log ""
log "Grid search launched. You can close your laptop."
log ""
log "  Live grid log     : ssh-in then 'tail -F /home/ubuntu/NFP_Predictor/_output_grid/grid_master_*.log'"
log "  Latest cell log   : ssh-in then 'tail -F /home/ubuntu/NFP_Predictor/_output_grid/cell_*/run.log'"
log "  Reattach tmux     : aws/ssh.sh   then: tmux a -t grid"
log ""
if [[ "${NO_AUTO_STOP_VAL}" == "1" ]]; then
  log "NO_AUTO_STOP=1 set, instance will keep running. Remember aws/stop.sh."
else
  log "When grid finishes (success OR failure), the instance will:"
  log "  1. sync _output_grid/ to s3://${S3_BUCKET:-<bucket>}/_output_grid/"
  log "  2. sudo shutdown -h +1 (auto-stop, EBS preserved)"
fi
